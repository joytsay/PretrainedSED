#include <QApplication>
#include <QAudioOutput>
#include <QCommandLineParser>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMediaPlayer>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QStatusBar>
#include <QUrl>
#include <QVBoxLayout>
#include <QVideoWidget>
#include <QVector>

#include <algorithm>
#include <cstdint>
#include <csignal>
#include <sys/types.h>
#include <utility>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kPacketMilliseconds = 40;
constexpr int kPacketSamples = kSampleRate * kPacketMilliseconds / 1000;
constexpr int kPacketBytes = kPacketSamples * 2;

struct ConfidenceFrame {
    qint64 timeMs = 0;
    QVector<double> scores;
};

class MainWindow final : public QMainWindow {
public:
    MainWindow(
        QString workerPath,
        QString enginePath,
        QString mappingPath,
        QString labelsPath,
        QString ffmpegPath,
        QString initialCamId
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
        auto* confidenceLayout = new QVBoxLayout(confidenceBox);
        auto* confidenceRowsWidget = new QWidget;
        confidenceRows_ = new QVBoxLayout(confidenceRowsWidget);
        confidenceRows_->setContentsMargins(0, 0, 0, 0);
        confidenceRows_->setAlignment(Qt::AlignTop);
        confidenceScroll_ = new QScrollArea;
        confidenceScroll_->setWidgetResizable(true);
        confidenceScroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        confidenceScroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        confidenceScroll_->setMinimumHeight(120);
        confidenceScroll_->setMaximumHeight(360);
        confidenceScroll_->setWidget(confidenceRowsWidget);
        confidenceLayout->addWidget(confidenceScroll_);
        auto* thresholdRow = new QHBoxLayout;
        thresholdRow->addWidget(new QLabel("Red threshold"));
        threshold_ = new QDoubleSpinBox;
        threshold_->setRange(0.0, 100.0);
        threshold_->setDecimals(1);
        threshold_->setSuffix("%");
        threshold_->setValue(10.0);
        thresholdRow->addWidget(threshold_);
        confidenceLayout->addLayout(thresholdRow);
        right->addWidget(confidenceBox);

        auto* cameraRow = new QHBoxLayout;
        cameraRow->addWidget(new QLabel("Camera ID"));
        camId_ = new QLineEdit(initialCamId);
        camId_->setPlaceholderText("camera-01");
        cameraRow->addWidget(camId_, 1);
        right->addLayout(cameraRow);

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

        ffmpegPath_ = std::move(ffmpegPath);
        decoder_ = new QProcess(this);
        decoder_->setProcessChannelMode(QProcess::SeparateChannels);
        connect(decoder_, &QProcess::readyReadStandardOutput, this, [this] { readDecodedAudio(); });
        connect(decoder_, &QProcess::readyReadStandardError, this, [this] {
            decoderErrorBuffer_ += decoder_->readAllStandardError();
        });
        connect(decoder_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
                [this](int exitCode, QProcess::ExitStatus status) {
            if (stoppingDecoder_) return;
            readDecodedAudio();
            finishDecodedAudio(exitCode, status);
        });
        connect(decoder_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
            if (stoppingDecoder_) return;
            workerStatus_->setText("FFmpeg error: " + decoder_->errorString());
            workerStatus_->setStyleSheet("color: red;");
        });

        connect(addButton_, &QPushButton::clicked, this, [this] { addFiles(); });
        connect(clearButton_, &QPushButton::clicked, this, [this] {
            player_->stop();
            stopDecoder(true);
            waitingForFirstResult_ = false;
            playButton_->setEnabled(true);
            playlist_->clear();
            files_.clear();
            frames_.clear();
            runButton_->setEnabled(false);
            resetScores();
        });
        connect(runButton_, &QPushButton::clicked, this, [this] {
            if (!workerReady_ || files_.isEmpty()) return;
            playIndex(0);
        });
        connect(playButton_, &QPushButton::clicked, this, [this] {
            if (waitingForFirstResult_) return;
            if (player_->playbackState() == QMediaPlayer::PlayingState) {
                player_->pause();
                setDecoderPaused(true);
            } else {
                if (decoder_->state() == QProcess::NotRunning &&
                    !restartCurrentAudioStream(player_->position(), true)) return;
                player_->play();
                setDecoderPaused(false);
            }
        });
        connect(stopButton_, &QPushButton::clicked, this, [this] {
            player_->stop();
            stopDecoder(true);
            waitingForFirstResult_ = false;
            playButton_->setEnabled(true);
        });
        connect(position_, &QSlider::sliderMoved, player_, &QMediaPlayer::setPosition);
        connect(position_, &QSlider::sliderPressed, this, [this] {
            seekWasPlaying_ = player_->playbackState() == QMediaPlayer::PlayingState;
            if (seekWasPlaying_) player_->pause();
            setDecoderPaused(true);
        });
        connect(position_, &QSlider::sliderReleased, this, [this] {
            restartCurrentAudioStream(player_->position(), seekWasPlaying_);
        });
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
        stopDecoder(true);
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
        playlist_->setCurrentRow(index);
        player_->setSource(QUrl::fromLocalFile(files_[index]));
        player_->setPosition(0);
        if (!restartCurrentAudioStream(0, true)) return;
        workerStatus_->setText(
            "Pre-rolling first timestamped result for " + QFileInfo(files_[index]).fileName() + "...");
        workerStatus_->setStyleSheet(QString());
    }

    bool restartCurrentAudioStream(qint64 startTimestampMs, bool playAfterFirstResult) {
        if (currentIndex_ < 0 || currentIndex_ >= files_.size() || !workerReady_) return false;
        const QString selectedCamId = camId_->text().trimmed();
        if (selectedCamId.isEmpty()) {
            QMessageBox::warning(this, "Missing camera ID", "Enter a camera ID before starting.");
            return false;
        }
        stopDecoder(true);
        ++currentRequest_;
        activeCamId_ = selectedCamId;
        frames_.clear();
        resetScores();
        waitingForFirstResult_ = true;
        playAfterFirstResult_ = playAfterFirstResult;
        playButton_->setEnabled(false);
        startAudioStream(files_[currentIndex_], startTimestampMs);
        if (!streamOpen_) {
            waitingForFirstResult_ = false;
            playButton_->setEnabled(true);
        }
        return streamOpen_;
    }

    void sendWorkerMessage(const QJsonObject& message) {
        if (worker_->state() != QProcess::Running) {
            workerStatus_->setText("Worker is not running");
            workerStatus_->setStyleSheet("color: red;");
            return;
        }
        const QByteArray line = QJsonDocument(message).toJson(QJsonDocument::Compact) + '\n';
        if (worker_->write(line) < 0) {
            workerStatus_->setText("Could not write audio packet to worker");
            workerStatus_->setStyleSheet("color: red;");
        }
    }

    void startAudioStream(const QString& path, qint64 startTimestampMs) {
        decoderBuffer_.clear();
        decoderErrorBuffer_.clear();
        nextAudioTimestampMs_ = startTimestampMs;
        decoderRequestId_ = currentRequest_;
        decoderCamId_ = activeCamId_;
        streamOpen_ = true;

        sendWorkerMessage({
            {"type", "stream_start"},
            {"id", static_cast<qint64>(decoderRequestId_)},
            {"cam_id", decoderCamId_},
            {"timestamp_ms", startTimestampMs},
        });

        QStringList arguments{
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-re",
        };
        if (startTimestampMs > 0) {
            arguments << "-ss" << QString::number(startTimestampMs / 1000.0, 'f', 3);
        }
        arguments << "-i" << path
                  << "-vn"
                  << "-map" << "0:a:0"
                  << "-ac" << "1"
                  << "-ar" << QString::number(kSampleRate)
                  << "-c:a" << "pcm_s16le"
                  << "-f" << "s16le"
                  << "pipe:1";
        decoder_->start(ffmpegPath_, arguments, QIODevice::ReadOnly);
        if (!decoder_->waitForStarted(1500)) {
            workerStatus_->setText("Could not start FFmpeg: " + decoder_->errorString());
            workerStatus_->setStyleSheet("color: red;");
            sendStreamEnd();
        }
    }

    void readDecodedAudio() {
        decoderBuffer_ += decoder_->readAllStandardOutput();
        while (decoderBuffer_.size() >= kPacketBytes && streamOpen_) {
            const QByteArray packet = decoderBuffer_.left(kPacketBytes);
            decoderBuffer_.remove(0, kPacketBytes);
            sendAudioPacket(packet);
        }
    }

    void sendAudioPacket(const QByteArray& packet) {
        if (!streamOpen_ || packet.size() != kPacketBytes) return;
        sendWorkerMessage({
            {"type", "audio"},
            {"id", static_cast<qint64>(decoderRequestId_)},
            {"cam_id", decoderCamId_},
            {"timestamp_ms", nextAudioTimestampMs_},
            {"sample_rate", kSampleRate},
            {"channels", 1},
            {"encoding", "s16le"},
            {"audio_b64", QString::fromLatin1(packet.toBase64())},
        });
        nextAudioTimestampMs_ += kPacketMilliseconds;
        if (worker_->bytesToWrite() > 4 * 1024 * 1024) {
            workerStatus_->setText("Worker is slower than the 40 ms audio stream; packets are queued");
            workerStatus_->setStyleSheet("color: #b06000; font-weight: 700;");
        }
    }

    void sendStreamEnd() {
        if (!streamOpen_) return;
        sendWorkerMessage({
            {"type", "stream_end"},
            {"id", static_cast<qint64>(decoderRequestId_)},
            {"cam_id", decoderCamId_},
            {"timestamp_ms", nextAudioTimestampMs_},
        });
        streamOpen_ = false;
    }

    void finishDecodedAudio(int exitCode, QProcess::ExitStatus status) {
        if (!streamOpen_) return;
        if (!decoderBuffer_.isEmpty()) {
            decoderBuffer_.append(QByteArray(kPacketBytes - decoderBuffer_.size(), '\0'));
            sendAudioPacket(decoderBuffer_);
            decoderBuffer_.clear();
        }
        sendStreamEnd();
        if (status != QProcess::NormalExit || exitCode != 0) {
            const QString detail = QString::fromUtf8(decoderErrorBuffer_).trimmed();
            workerStatus_->setText("FFmpeg extraction failed" +
                                   (detail.isEmpty() ? QString() : ": " + detail));
            workerStatus_->setStyleSheet("color: red;");
        }
    }

    void stopDecoder(bool closeStream) {
        stoppingDecoder_ = true;
        if (decoder_ && decoder_->state() != QProcess::NotRunning) {
            const qint64 processId = decoder_->processId();
            if (processId > 0) ::kill(static_cast<pid_t>(processId), SIGCONT);
            decoder_->terminate();
            if (!decoder_->waitForFinished(1000)) {
                decoder_->kill();
                decoder_->waitForFinished(1000);
            }
        }
        stoppingDecoder_ = false;
        decoderBuffer_.clear();
        decoderErrorBuffer_.clear();
        if (closeStream) sendStreamEnd();
    }

    void setDecoderPaused(bool paused) {
        if (!decoder_ || decoder_->state() != QProcess::Running) return;
        const qint64 processId = decoder_->processId();
        if (processId <= 0) return;
        ::kill(static_cast<pid_t>(processId), paused ? SIGSTOP : SIGCONT);
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
            if (classes.isEmpty()) {
                workerStatus_->setText("Worker mapping did not return any classes");
                workerStatus_->setStyleSheet("color: red;");
                return;
            }
            configureClasses(classes);
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
        if (event == "stream_started") {
            workerStatus_->setText(
                "Receiving 40 ms packets for " + activeCamId_ + " (history prefilled with silence)...");
            workerStatus_->setStyleSheet(QString());
        } else if (event == "buffering") {
            const qint64 bufferedMs = callback.value("buffered_ms").toInteger();
            workerStatus_->setText(
                QString("Buffering %1 / 10000 ms for %2...").arg(bufferedMs).arg(activeCamId_));
            workerStatus_->setStyleSheet(QString());
        } else if (event == "result") {
            const qint64 timestampMs = callback.value("timestamp_ms").toInteger();
            const QJsonArray scores = callback.value("scores").toArray();
            if (scores.size() == classLabels_.size()) {
                QVector<double> values;
                values.reserve(scores.size());
                for (const QJsonValue score : scores) values.push_back(score.toDouble());
                frames_.push_back({timestampMs, std::move(values)});
            } else {
                workerStatus_->setText(
                    QString("Worker returned %1 scores for %2 mapped classes")
                        .arg(scores.size()).arg(classLabels_.size()));
                workerStatus_->setStyleSheet("color: red;");
                return;
            }
            const qint64 processingMs = callback.value("processing_ms").toInteger();
            const qint64 superseded = callback.value("superseded_packets").toInteger();
            if (waitingForFirstResult_) {
                waitingForFirstResult_ = false;
                playButton_->setEnabled(true);
                player_->setPosition(timestampMs);
                if (playAfterFirstResult_) {
                    player_->play();
                    setDecoderPaused(false);
                } else {
                    setDecoderPaused(true);
                }
            }
            const qint64 playbackLagMs = std::max<qint64>(0, player_->position() - timestampMs);
            workerStatus_->setText(
                QString("Live result for %1 at %2 ms (inference %3 ms, playback lag %4 ms, "
                        "superseded %5)")
                    .arg(activeCamId_).arg(timestampMs).arg(processingMs)
                    .arg(playbackLagMs).arg(superseded));
            workerStatus_->setStyleSheet(
                playbackLagMs > 200 ? "color: #b06000; font-weight: 700;" :
                                      "color: #198754; font-weight: 700;");
            updateAtPosition(player_->position());
        } else if (event == "complete") {
            if (waitingForFirstResult_) {
                waitingForFirstResult_ = false;
                playButton_->setEnabled(true);
                if (playAfterFirstResult_) player_->play();
            }
            workerStatus_->setText("Audio stream complete");
            workerStatus_->setStyleSheet("color: #198754; font-weight: 700;");
        } else if (event == "error") {
            workerStatus_->setText("Analysis error: " + callback.value("message").toString());
            workerStatus_->setStyleSheet("color: red;");
        }
    }

    void updateAtPosition(qint64 position) {
        if (frames_.isEmpty()) {
            resetScores();
            return;
        }
        const auto iterator = std::upper_bound(
            frames_.cbegin(), frames_.cend(), position,
            [](qint64 time, const ConfidenceFrame& frame) { return time < frame.timeMs; }
        );
        const auto selected = iterator == frames_.cbegin() ? iterator : std::prev(iterator);
        if (selected != frames_.cend()) showScores(selected->scores);
    }

    void configureClasses(const QJsonArray& classes) {
        for (QWidget* row : classRowWidgets_) {
            confidenceRows_->removeWidget(row);
            row->deleteLater();
        }
        classRowWidgets_.clear();
        classLabels_.clear();
        confidenceLabels_.clear();
        displayOrder_.clear();

        for (const QJsonValue value : classes) {
            auto* row = new QWidget;
            row->setMinimumHeight(40);
            auto* layout = new QHBoxLayout(row);
            layout->setContentsMargins(0, 0, 0, 0);
            auto* classLabel = new QLabel(value.toString());
            classLabel->setStyleSheet("font-size: 24px; font-weight: 700;");
            auto* confidenceLabel = new QLabel("0.0%");
            confidenceLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
            confidenceLabel->setStyleSheet("font-size: 24px; font-weight: 700;");
            layout->addWidget(classLabel, 1);
            layout->addWidget(confidenceLabel);
            confidenceRows_->addWidget(row);
            classRowWidgets_.push_back(row);
            classLabels_.push_back(classLabel);
            confidenceLabels_.push_back(confidenceLabel);
            displayOrder_.push_back(displayOrder_.size());
        }
        const int visibleRows = std::clamp(static_cast<int>(classes.size()), 1, 8);
        const int panelHeight = visibleRows * 44 + 8;
        confidenceScroll_->setMinimumHeight(panelHeight);
        confidenceScroll_->setMaximumHeight(panelHeight);
    }

    void resetScores() {
        showScores(QVector<double>(classLabels_.size(), 0.0));
    }

    void showScores(const QVector<double>& scores) {
        const double threshold = threshold_->value() / 100.0;
        for (qsizetype index = 0; index < classLabels_.size(); ++index) {
            const double score = index < scores.size() ? scores[index] : 0.0;
            const bool alarm = score > threshold;
            const QString color = alarm ? "#d00000" : "palette(text)";
            classLabels_[index]->setStyleSheet(
                QString("font-size: 24px; font-weight: 700; color: %1;").arg(color));
            confidenceLabels_[index]->setStyleSheet(
                QString("font-size: 24px; font-weight: 700; color: %1;").arg(color));
            confidenceLabels_[index]->setText(QString::number(score * 100.0, 'f', 1) + "%");
        }

        QVector<int> sortedOrder;
        sortedOrder.reserve(classLabels_.size());
        for (int index = 0; index < classLabels_.size(); ++index) sortedOrder.push_back(index);
        std::stable_sort(sortedOrder.begin(), sortedOrder.end(), [&scores](int left, int right) {
            const double leftScore = left < scores.size() ? scores[left] : 0.0;
            const double rightScore = right < scores.size() ? scores[right] : 0.0;
            return leftScore > rightScore;
        });
        if (sortedOrder != displayOrder_) {
            for (QWidget* row : classRowWidgets_) confidenceRows_->removeWidget(row);
            for (const int index : sortedOrder) confidenceRows_->addWidget(classRowWidgets_[index]);
            displayOrder_ = std::move(sortedOrder);
        }
    }

    QMediaPlayer* player_ = nullptr;
    QAudioOutput* audio_ = nullptr;
    QVideoWidget* videoWidget_ = nullptr;
    QPushButton* playButton_ = nullptr;
    QPushButton* stopButton_ = nullptr;
    QSlider* position_ = nullptr;
    QLabel* timeLabel_ = nullptr;
    QScrollArea* confidenceScroll_ = nullptr;
    QVBoxLayout* confidenceRows_ = nullptr;
    QVector<QWidget*> classRowWidgets_;
    QVector<QLabel*> classLabels_;
    QVector<QLabel*> confidenceLabels_;
    QVector<int> displayOrder_;
    QDoubleSpinBox* threshold_ = nullptr;
    QLineEdit* camId_ = nullptr;
    QPushButton* addButton_ = nullptr;
    QPushButton* clearButton_ = nullptr;
    QPushButton* runButton_ = nullptr;
    QListWidget* playlist_ = nullptr;
    QLabel* workerStatus_ = nullptr;
    QProcess* worker_ = nullptr;
    QProcess* decoder_ = nullptr;
    QByteArray callbackBuffer_;
    QByteArray decoderBuffer_;
    QByteArray decoderErrorBuffer_;
    QString ffmpegPath_;
    QString activeCamId_;
    QString decoderCamId_;
    QStringList files_;
    QVector<ConfidenceFrame> frames_;
    int currentIndex_ = -1;
    qint64 currentRequest_ = 0;
    qint64 decoderRequestId_ = -1;
    qint64 nextAudioTimestampMs_ = 0;
    bool workerReady_ = false;
    bool streamOpen_ = false;
    bool stoppingDecoder_ = false;
    bool seekWasPlaying_ = false;
    bool waitingForFirstResult_ = false;
    bool playAfterFirstResult_ = true;
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
    parser.addOption({"mapping", "Path to the aggregate class mapping CSV", "path",
                      "/workspace/class_mapping.csv"});
    parser.addOption({"labels", "Path to the ordered 447-class label file", "path",
                      "/workspace/resources/ATST-F_strong_1.labels.txt"});
    parser.addOption({"ffmpeg", "Path to FFmpeg used for 40 ms PCM extraction", "path", "ffmpeg"});
    parser.addOption({"cam-id", "Initial camera/source identifier", "id", "camera-01"});
    parser.process(application);

    MainWindow window(
        parser.value("worker"),
        parser.value("engine"),
        parser.value("mapping"),
        parser.value("labels"),
        parser.value("ffmpeg"),
        parser.value("cam-id")
    );
    window.show();
    return application.exec();
}
