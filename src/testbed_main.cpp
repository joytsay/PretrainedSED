#include <QApplication>
#include <QAudioOutput>
#include <QCommandLineParser>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QMediaPlayer>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QSlider>
#include <QStatusBar>
#include <QUrl>
#include <QVBoxLayout>
#include <QVideoWidget>

#include <algorithm>
#include <array>
#include <cstdint>

namespace {

struct ConfidenceFrame {
    qint64 timeMs = 0;
    std::array<double, 3> scores{};
};

class MainWindow final : public QMainWindow {
public:
    MainWindow(
        QString workerPath,
        QString enginePath,
        QString mappingPath,
        QString labelsPath
    ) {
        setWindowTitle("ATST-F Aggregate SED Testbed");
        resize(1180, 760);

        auto* central = new QWidget;
        auto* root = new QHBoxLayout(central);
        auto* left = new QVBoxLayout;
        auto* right = new QVBoxLayout;
        root->addLayout(left, 2);
        root->addLayout(right, 1);
        setCentralWidget(central);

        videoWidget_ = new QVideoWidget;
        videoWidget_->setMinimumSize(720, 405);
        left->addWidget(videoWidget_, 1);

        auto* transport = new QHBoxLayout;
        playButton_ = new QPushButton("Play");
        stopButton_ = new QPushButton("Stop");
        position_ = new QSlider(Qt::Horizontal);
        position_->setRange(0, 0);
        timeLabel_ = new QLabel("00:00 / 00:00");
        transport->addWidget(playButton_);
        transport->addWidget(stopButton_);
        transport->addWidget(position_, 1);
        transport->addWidget(timeLabel_);
        left->addLayout(transport);

        auto* confidenceBox = new QGroupBox("Aggregate confidence");
        auto* confidenceLayout = new QGridLayout(confidenceBox);
        const std::array<QString, 3> names{"Class 1", "Class 2", "Class 3"};
        for (int index = 0; index < 3; ++index) {
            classLabels_[index] = new QLabel(names[index]);
            classLabels_[index]->setStyleSheet("font-size: 24px; font-weight: 700;");
            confidenceLabels_[index] = new QLabel("0.0%");
            confidenceLabels_[index]->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
            confidenceLabels_[index]->setStyleSheet("font-size: 24px; font-weight: 700;");
            confidenceLayout->addWidget(classLabels_[index], index, 0);
            confidenceLayout->addWidget(confidenceLabels_[index], index, 1);
        }
        confidenceLayout->addWidget(new QLabel("Red threshold"), 3, 0);
        threshold_ = new QDoubleSpinBox;
        threshold_->setRange(0.0, 100.0);
        threshold_->setDecimals(1);
        threshold_->setSuffix("%");
        threshold_->setValue(50.0);
        confidenceLayout->addWidget(threshold_, 3, 1);
        right->addWidget(confidenceBox);

        auto* pickerRow = new QHBoxLayout;
        addButton_ = new QPushButton("Add media files...");
        clearButton_ = new QPushButton("Clear");
        pickerRow->addWidget(addButton_);
        pickerRow->addWidget(clearButton_);
        right->addLayout(pickerRow);
        playlist_ = new QListWidget;
        playlist_->setSelectionMode(QAbstractItemView::SingleSelection);
        right->addWidget(playlist_, 1);
        runButton_ = new QPushButton("Run playlist");
        runButton_->setEnabled(false);
        right->addWidget(runButton_);

        workerStatus_ = new QLabel("Starting callback worker...");
        workerStatus_->setWordWrap(true);
        right->addWidget(workerStatus_);

        player_ = new QMediaPlayer(this);
        audio_ = new QAudioOutput(this);
        player_->setAudioOutput(audio_);
        player_->setVideoOutput(videoWidget_);
        audio_->setVolume(0.8F);

        worker_ = new QProcess(this);
        worker_->setProcessChannelMode(QProcess::SeparateChannels);
        connect(worker_, &QProcess::readyReadStandardOutput, this, [this] { readCallbacks(); });
        connect(worker_, &QProcess::readyReadStandardError, this, [this] {
            const QString message = QString::fromUtf8(worker_->readAllStandardError()).trimmed();
            if (!message.isEmpty()) statusBar()->showMessage(message, 8000);
        });
        connect(worker_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
            workerStatus_->setText("Worker error: " + worker_->errorString());
            workerStatus_->setStyleSheet("color: red;");
        });

        connect(addButton_, &QPushButton::clicked, this, [this] { addFiles(); });
        connect(clearButton_, &QPushButton::clicked, this, [this] {
            player_->stop();
            playlist_->clear();
            files_.clear();
            frames_.clear();
            runButton_->setEnabled(false);
            showScores({0.0, 0.0, 0.0});
        });
        connect(runButton_, &QPushButton::clicked, this, [this] {
            if (!workerReady_ || files_.isEmpty()) return;
            playIndex(0);
        });
        connect(playButton_, &QPushButton::clicked, this, [this] {
            if (player_->playbackState() == QMediaPlayer::PlayingState) player_->pause();
            else player_->play();
        });
        connect(stopButton_, &QPushButton::clicked, player_, &QMediaPlayer::stop);
        connect(position_, &QSlider::sliderMoved, player_, &QMediaPlayer::setPosition);
        connect(player_, &QMediaPlayer::positionChanged, this, [this](qint64 position) {
            if (!position_->isSliderDown()) position_->setValue(static_cast<int>(position));
            updateAtPosition(position);
            updateTime(position, player_->duration());
        });
        connect(player_, &QMediaPlayer::durationChanged, this, [this](qint64 duration) {
            position_->setRange(0, static_cast<int>(std::min<qint64>(duration, INT_MAX)));
            updateTime(player_->position(), duration);
        });
        connect(player_, &QMediaPlayer::playbackStateChanged, this, [this](QMediaPlayer::PlaybackState state) {
            playButton_->setText(state == QMediaPlayer::PlayingState ? "Pause" : "Play");
        });
        connect(player_, &QMediaPlayer::mediaStatusChanged, this, [this](QMediaPlayer::MediaStatus status) {
            if (status == QMediaPlayer::EndOfMedia && currentIndex_ + 1 < files_.size()) {
                playIndex(currentIndex_ + 1);
            }
        });
        connect(threshold_, &QDoubleSpinBox::valueChanged, this, [this] { updateAtPosition(player_->position()); });
        connect(playlist_, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
            playIndex(playlist_->row(item));
        });

        worker_->start(workerPath, {enginePath, mappingPath, labelsPath});
    }

    ~MainWindow() override {
        worker_->closeWriteChannel();
        if (!worker_->waitForFinished(1500)) {
            worker_->terminate();
            worker_->waitForFinished(1000);
        }
    }

private:
    static QString clockText(qint64 milliseconds) {
        const qint64 seconds = std::max<qint64>(0, milliseconds / 1000);
        return QString("%1:%2").arg(seconds / 60, 2, 10, QLatin1Char('0'))
                                 .arg(seconds % 60, 2, 10, QLatin1Char('0'));
    }

    void updateTime(qint64 position, qint64 duration) {
        timeLabel_->setText(clockText(position) + " / " + clockText(duration));
    }

    void addFiles() {
        const QStringList selected = QFileDialog::getOpenFileNames(
            this,
            "Choose audio/video playlist",
            QDir::homePath(),
            "Media (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.wav *.flac *.mp3 *.ogg *.m4a *.aac);;All files (*)"
        );
        for (const QString& path : selected) {
            if (files_.contains(path)) continue;
            files_.push_back(path);
            auto* item = new QListWidgetItem(QFileInfo(path).fileName());
            item->setToolTip(path);
            playlist_->addItem(item);
        }
        runButton_->setEnabled(workerReady_ && !files_.isEmpty());
    }

    void playIndex(int index) {
        if (index < 0 || index >= files_.size() || !workerReady_) return;
        currentIndex_ = index;
        currentRequest_++;
        frames_.clear();
        playlist_->setCurrentRow(index);
        showScores({0.0, 0.0, 0.0});
        const QJsonObject request{
            {"id", static_cast<qint64>(currentRequest_)},
            {"path", files_[index]},
        };
        worker_->write(QJsonDocument(request).toJson(QJsonDocument::Compact) + '\n');
        player_->setSource(QUrl::fromLocalFile(files_[index]));
        player_->play();
        workerStatus_->setText("Analyzing " + QFileInfo(files_[index]).fileName() + "...");
        workerStatus_->setStyleSheet(QString());
    }

    void readCallbacks() {
        callbackBuffer_ += worker_->readAllStandardOutput();
        while (true) {
            const qsizetype newline = callbackBuffer_.indexOf('\n');
            if (newline < 0) break;
            const QByteArray line = callbackBuffer_.left(newline).trimmed();
            callbackBuffer_.remove(0, newline + 1);
            if (line.isEmpty()) continue;
            QJsonParseError error;
            const QJsonDocument document = QJsonDocument::fromJson(line, &error);
            if (error.error != QJsonParseError::NoError || !document.isObject()) {
                statusBar()->showMessage("Invalid worker callback: " + QString::fromUtf8(line), 8000);
                continue;
            }
            handleCallback(document.object());
        }
    }

    void handleCallback(const QJsonObject& callback) {
        const QString event = callback.value("event").toString();
        if (event == "ready") {
            const QJsonArray classes = callback.value("classes").toArray();
            if (classes.size() != 3) {
                workerStatus_->setText("Worker mapping did not return exactly three classes");
                workerStatus_->setStyleSheet("color: red;");
                return;
            }
            for (int index = 0; index < 3; ++index) {
                classLabels_[index]->setText(classes[index].toString());
            }
            workerReady_ = true;
            workerStatus_->setText("TensorRT callback worker ready");
            workerStatus_->setStyleSheet("color: #198754; font-weight: 700;");
            runButton_->setEnabled(!files_.isEmpty());
            return;
        }
        if (event == "fatal") {
            workerStatus_->setText("Fatal worker error: " + callback.value("message").toString());
            workerStatus_->setStyleSheet("color: red;");
            return;
        }
        const qint64 id = callback.value("id").toInteger(-1);
        if (id != currentRequest_) return;
        if (event == "chunk") {
            const qint64 start = callback.value("start_ms").toInteger();
            const qint64 hop = callback.value("hop_ms").toInteger(40);
            const QJsonArray scores = callback.value("scores").toArray();
            frames_.reserve(frames_.size() + scores.size());
            for (qsizetype index = 0; index < scores.size(); ++index) {
                const QJsonArray row = scores[index].toArray();
                if (row.size() != 3) continue;
                frames_.push_back({
                    start + static_cast<qint64>(index) * hop,
                    {row[0].toDouble(), row[1].toDouble(), row[2].toDouble()},
                });
            }
            updateAtPosition(player_->position());
        } else if (event == "complete") {
            workerStatus_->setText("Analysis complete");
            workerStatus_->setStyleSheet("color: #198754; font-weight: 700;");
        } else if (event == "error") {
            workerStatus_->setText("Analysis error: " + callback.value("message").toString());
            workerStatus_->setStyleSheet("color: red;");
        }
    }

    void updateAtPosition(qint64 position) {
        if (frames_.isEmpty()) {
            showScores({0.0, 0.0, 0.0});
            return;
        }
        const auto iterator = std::upper_bound(
            frames_.cbegin(), frames_.cend(), position,
            [](qint64 time, const ConfidenceFrame& frame) { return time < frame.timeMs; }
        );
        const auto selected = iterator == frames_.cbegin() ? iterator : std::prev(iterator);
        if (selected != frames_.cend()) showScores(selected->scores);
    }

    void showScores(const std::array<double, 3>& scores) {
        const double threshold = threshold_->value() / 100.0;
        for (int index = 0; index < 3; ++index) {
            const bool alarm = scores[index] > threshold;
            const QString color = alarm ? "#d00000" : "palette(text)";
            classLabels_[index]->setStyleSheet(
                QString("font-size: 24px; font-weight: 700; color: %1;").arg(color));
            confidenceLabels_[index]->setStyleSheet(
                QString("font-size: 24px; font-weight: 700; color: %1;").arg(color));
            confidenceLabels_[index]->setText(QString::number(scores[index] * 100.0, 'f', 1) + "%");
        }
    }

    QMediaPlayer* player_ = nullptr;
    QAudioOutput* audio_ = nullptr;
    QVideoWidget* videoWidget_ = nullptr;
    QPushButton* playButton_ = nullptr;
    QPushButton* stopButton_ = nullptr;
    QSlider* position_ = nullptr;
    QLabel* timeLabel_ = nullptr;
    std::array<QLabel*, 3> classLabels_{};
    std::array<QLabel*, 3> confidenceLabels_{};
    QDoubleSpinBox* threshold_ = nullptr;
    QPushButton* addButton_ = nullptr;
    QPushButton* clearButton_ = nullptr;
    QPushButton* runButton_ = nullptr;
    QListWidget* playlist_ = nullptr;
    QLabel* workerStatus_ = nullptr;
    QProcess* worker_ = nullptr;
    QByteArray callbackBuffer_;
    QStringList files_;
    QVector<ConfidenceFrame> frames_;
    int currentIndex_ = -1;
    qint64 currentRequest_ = 0;
    bool workerReady_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    QApplication application(argc, argv);
    QCoreApplication::setApplicationName("ATST-F SED Testbed");
    QCommandLineParser parser;
    parser.setApplicationDescription("Three-class TensorRT ATST-F playlist testbed");
    parser.addHelpOption();
    parser.addOption({"worker", "Path to atst_sed_worker", "path", "atst_sed_worker"});
    parser.addOption({"engine", "Path to ATST-F TensorRT engine", "path",
                      "/workspace/resources/ATST-F_strong_1.trt"});
    parser.addOption({"mapping", "Path to the three-class aggregation CSV", "path",
                      "/workspace/class_mapping.csv"});
    parser.addOption({"labels", "Path to the ordered 447-class label file", "path",
                      "/workspace/resources/ATST-F_strong_1.labels.txt"});
    parser.process(application);

    MainWindow window(
        parser.value("worker"),
        parser.value("engine"),
        parser.value("mapping"),
        parser.value("labels")
    );
    window.show();
    return application.exec();
}
