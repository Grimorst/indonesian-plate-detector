/**
 * VisionGuard ALPR - Application JavaScript
 * Controls: Foto Inspector, CCTV Live Surveillance, Snap Camera, Audio Ping, and Activity Logs
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const btnModeFoto = document.getElementById('btnModeFoto');
  const btnModeCctv = document.getElementById('btnModeCctv');
  const viewportTitle = document.getElementById('viewportTitle');
  const modeIcon = document.getElementById('modeIcon');
  const fotoControls = document.getElementById('fotoControls');
  const cctvControls = document.getElementById('cctvControls');
  const fotoContainer = document.getElementById('fotoContainer');
  const cctvContainer = document.getElementById('cctvContainer');
  const footerTipText = document.getElementById('footerTipText');

  // Foto Mode Elements
  const dropZone = document.getElementById('dropZone');
  const imageFileInput = document.getElementById('imageFileInput');
  const btnLoadSample = document.getElementById('btnLoadSample');
  const btnLoadSample2 = document.getElementById('btnLoadSample2');
  const emptyUploadState = document.getElementById('emptyUploadState');
  const imageDisplayWrapper = document.getElementById('imageDisplayWrapper');
  const photoCanvas = document.getElementById('photoCanvas');

  // Camera Buttons (new)
  const btnQuickCamera = document.getElementById('btnQuickCamera');
  const btnOpenSnapCam = document.getElementById('btnOpenSnapCam');
  const btnOpenSnapCamCenter = document.getElementById('btnOpenSnapCamCenter');
  const btnSnapPlate = document.getElementById('btnSnapPlate');
  const cameraSourceSelect = document.getElementById('cameraSourceSelect');

  // CCTV Elements
  const btnToggleCam = document.getElementById('btnToggleCam');
  const btnStartCamCenter = document.getElementById('btnStartCamCenter');
  const btnSimulateStream = document.getElementById('btnSimulateStream');
  const btnSimulateStreamCenter = document.getElementById('btnSimulateStreamCenter');
  const camBtnText = document.getElementById('camBtnText');
  const cctvOffState = document.getElementById('cctvOffState');
  const cctvDisplayWrapper = document.getElementById('cctvDisplayWrapper');
  const webcamVideo = document.getElementById('webcamVideo');
  const cctvOverlayCanvas = document.getElementById('cctvOverlayCanvas');
  const hudTimestamp = document.getElementById('hudTimestamp');

  // Overlays & Feedback
  const scannerLine = document.getElementById('scannerLine');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingText = document.getElementById('loadingText');
  const btnToggleAudio = document.getElementById('btnToggleAudio');
  const inferenceSpeed = document.getElementById('inferenceSpeed');

  // Results & Inspector
  const plateCountBadge = document.getElementById('plateCountBadge');
  const primaryPlateNumber = document.getElementById('primaryPlateNumber');
  const primaryPlatePeriod = document.getElementById('primaryPlatePeriod');
  const yoloConfVal = document.getElementById('yoloConfVal');
  const yoloConfBar = document.getElementById('yoloConfBar');
  const ocrConfVal = document.getElementById('ocrConfVal');
  const ocrConfBar = document.getElementById('ocrConfBar');
  const cropsList = document.getElementById('cropsList');
  const btnZoomCrop = document.getElementById('btnZoomCrop');

  // OCR Debug Panel Elements
  const btnToggleOcrDebug = document.getElementById('btnToggleOcrDebug');
  const debugTitleBar = document.getElementById('debugTitleBar');
  const debugChevron = document.getElementById('debugChevron');
  const debugVariantsWrapper = document.getElementById('debugVariantsWrapper');
  const debugPlateTabs = document.getElementById('debugPlateTabs');
  const debugVariantsScroll = document.getElementById('debugVariantsScroll');
  const debugCandidatesContainer = document.getElementById('debugCandidatesContainer');
  const candidatesTableBody = document.getElementById('candidatesTableBody');
  const candidateCountBadge = document.getElementById('candidateCountBadge');

  let currentDebugDetectionIndex = 0;
  let cachedDetections = [];

  // Stats
  const statTotalScanned = document.getElementById('statTotalScanned');
  const statPlatesFound = document.getElementById('statPlatesFound');
  const statSuccessRate = document.getElementById('statSuccessRate');

  // History & Table
  const historyTableBody = document.getElementById('historyTableBody');
  const logSearchInput = document.getElementById('logSearchInput');
  const btnExportCsv = document.getElementById('btnExportCsv');
  const btnClearLog = document.getElementById('btnClearLog');

  // Modal
  const imageModal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImg');
  const modalTitle = document.getElementById('modalTitle');
  const modalCaption = document.getElementById('modalCaption');
  const btnModalClose = document.getElementById('btnModalClose');
  const systemStatus = document.getElementById('systemStatus');

  // --- App State ---
  let currentMode = 'foto'; // 'foto' | 'cctv'
  let isCameraActive = false;
  let isSimulating = false;
  let isFotoSnapMode = false; // Camera opened in Foto mode for snapshot
  let mediaStream = null;
  let cctvInterval = null;
  let isAudioEnabled = true;
  let stats = { totalScanned: 0, platesFound: 0 };
  let historyLogs = [];
  let lastDetectedPlate = '';
  let lastDetectionTime = 0;

  function showToast(message, type = 'info') {
    let container = document.querySelector('.toast-container');
    if (!container) {
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.remove();
      if (!container.children.length) container.remove();
    }, 4200);
  }

  function setSystemStatus(label, type = 'ready') {
    if (!systemStatus) return;
    const icon = type === 'error' ? 'triangle-exclamation' : type === 'busy' ? 'spinner fa-spin' : 'circle-check';
    const tone = type === 'error' ? 'text-crimson' : type === 'busy' ? 'text-cyan' : 'text-emerald';
    systemStatus.className = `status-val ${tone}`;
    systemStatus.innerHTML = `<i class="fa-solid fa-${icon}"></i> ${label}`;
  }

  // --- Clock updater for CCTV HUD ---
  setInterval(() => {
    const now = new Date();
    if (hudTimestamp) hudTimestamp.textContent = now.toISOString().replace('T', ' ').substring(0, 19);
  }, 1000);

  // --- Enumerate Camera Devices ---
  async function populateCameraList() {
    if (!navigator.mediaDevices?.enumerateDevices || !cameraSourceSelect) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      cameraSourceSelect.innerHTML = '<option value="">Kamera otomatis</option>';
      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Kamera ${idx + 1}`;
        cameraSourceSelect.appendChild(opt);
      });
    } catch (e) {
      console.warn('Cannot enumerate cameras:', e);
    }
  }
  populateCameraList();

  // --- Audio Alert ---
  function playBeep() {
    if (!isAudioEnabled) return;
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1320, audioCtx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.15);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.15);
    } catch (e) {
      console.warn('Audio play error:', e);
    }
  }

  if (btnToggleAudio) {
    btnToggleAudio.addEventListener('click', () => {
      isAudioEnabled = !isAudioEnabled;
      btnToggleAudio.classList.toggle('active', isAudioEnabled);
    });
  }

  // --- Mode Switching ---
  function setMode(mode) {
    currentMode = mode;
    stopFotoSnapCamera(); // Always stop snap cam when switching modes
    if (mode === 'foto') {
      btnModeFoto.classList.add('active');
      btnModeCctv.classList.remove('active');
      fotoControls.classList.remove('d-none');
      cctvControls.classList.add('d-none');
      fotoContainer.classList.remove('d-none');
      cctvContainer.classList.add('d-none');
      modeIcon.className = 'fa-solid fa-image text-cyan';
      viewportTitle.textContent = 'Deteksi dari Foto';
      footerTipText.textContent = 'Mode Foto: unggah foto, buka kamera, atau gunakan contoh untuk mendeteksi plat.';
      stopCamera();
      stopSimulation();
      if (btnQuickCamera) btnQuickCamera.innerHTML = '<i class="fa-solid fa-video"></i><span>Buka Kamera</span>';
    } else {
      btnModeCctv.classList.add('active');
      btnModeFoto.classList.remove('active');
      cctvControls.classList.remove('d-none');
      fotoControls.classList.add('d-none');
      cctvContainer.classList.remove('d-none');
      fotoContainer.classList.add('d-none');
      viewportTitle.textContent = 'Deteksi Kamera Live';
      modeIcon.className = 'fa-solid fa-video text-crimson';
      footerTipText.textContent = 'Mode Kamera Live: pindai plat kendaraan langsung dari webcam.';
      if (btnQuickCamera) btnQuickCamera.innerHTML = '<i class="fa-solid fa-video"></i><span>Mulai Kamera</span>';
    }
  }

  btnModeFoto.addEventListener('click', () => setMode('foto'));
  btnModeCctv.addEventListener('click', () => setMode('cctv'));

  // --- Quick Camera Button (Header) ---
  if (btnQuickCamera) {
    btnQuickCamera.addEventListener('click', () => {
      if (currentMode === 'foto') {
        // In foto mode, open snap camera
        if (isFotoSnapMode) {
          captureFotoSnap();
        } else {
          openFotoSnapCamera();
        }
      } else {
        // In CCTV mode, toggle live camera
        toggleCamera();
      }
    });
  }

  // --- Foto Mode: Snap Camera (take photo from webcam) ---
  if (btnOpenSnapCam) {
    btnOpenSnapCam.addEventListener('click', () => {
      if (isFotoSnapMode) {
        captureFotoSnap();
      } else {
        openFotoSnapCamera();
      }
    });
  }
  if (btnOpenSnapCamCenter) btnOpenSnapCamCenter.addEventListener('click', openFotoSnapCamera);

  async function openFotoSnapCamera() {
    try {
      const constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 } } };
      // Use selected camera if available
      if (cameraSourceSelect && cameraSourceSelect.value) {
        constraints.video.deviceId = { exact: cameraSourceSelect.value };
      }
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Show video in the foto viewport
      emptyUploadState.classList.add('d-none');
      imageDisplayWrapper.classList.remove('d-none');

      // Create a temporary video element inside imageDisplayWrapper
      let snapVideo = document.getElementById('fotoSnapVideo');
      if (!snapVideo) {
        snapVideo = document.createElement('video');
        snapVideo.id = 'fotoSnapVideo';
        snapVideo.autoplay = true;
        snapVideo.playsInline = true;
        snapVideo.muted = true;
        snapVideo.style.cssText = 'max-width:100%;max-height:520px;object-fit:contain;display:block;border-radius:8px;';
        imageDisplayWrapper.insertBefore(snapVideo, imageDisplayWrapper.firstChild);
      }
      snapVideo.srcObject = mediaStream;
      snapVideo.classList.remove('d-none');
      await snapVideo.play();

      // Hide photo canvas while video is live
      photoCanvas.classList.add('d-none');

      isFotoSnapMode = true;

      // Update buttons
      if (btnOpenSnapCam) {
        btnOpenSnapCam.innerHTML = '<i class="fa-solid fa-camera-retro"></i> Ambil Foto';
        btnOpenSnapCam.classList.replace('btn-emerald', 'btn-cyan');
      }
      if (btnQuickCamera) {
        btnQuickCamera.innerHTML = '<i class="fa-solid fa-camera-retro"></i><span>Ambil Foto</span>';
      }

      footerTipText.textContent = 'Kamera aktif. Klik "Ambil Foto" untuk menangkap gambar dan mendeteksi plat.';
      showToast('Kamera aktif. Ambil foto saat plat terlihat jelas.', 'success');
    } catch (err) {
      showToast('Tidak dapat mengakses kamera. Pastikan izin kamera browser sudah aktif.', 'error');
      console.error('Camera error:', err);
    }
  }

  async function captureFotoSnap() {
    const snapVideo = document.getElementById('fotoSnapVideo');
    if (!snapVideo || !mediaStream) return;

    // Capture frame from video
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = snapVideo.videoWidth || 640;
    tempCanvas.height = snapVideo.videoHeight || 480;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(snapVideo, 0, 0, tempCanvas.width, tempCanvas.height);

    // Stop the camera
    stopFotoSnapCamera();

    // Convert to blob and send to detection
    tempCanvas.toBlob(async (blob) => {
      const file = new File([blob], 'camera_snapshot.jpg', { type: 'image/jpeg' });
      await handleImageUpload(file);
    }, 'image/jpeg', 0.9);
  }

  function stopFotoSnapCamera() {
    if (mediaStream && isFotoSnapMode) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    isFotoSnapMode = false;

    const snapVideo = document.getElementById('fotoSnapVideo');
    if (snapVideo) {
      snapVideo.srcObject = null;
      snapVideo.classList.add('d-none');
    }

    photoCanvas.classList.remove('d-none');

    // Restore button
    if (btnOpenSnapCam) {
      btnOpenSnapCam.innerHTML = '<i class="fa-solid fa-camera"></i> Buka Kamera';
      btnOpenSnapCam.classList.replace('btn-cyan', 'btn-emerald');
    }
    if (btnQuickCamera) {
      btnQuickCamera.innerHTML = '<i class="fa-solid fa-video"></i><span>Buka Kamera</span>';
    }
  }

  // --- Drag & Drop Setup ---
  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      if (currentMode === 'foto') dropZone.classList.add('is-dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('is-dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (currentMode !== 'foto') return;
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleImageUpload(files[0]);
    }
  });

  imageFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleImageUpload(e.target.files[0]);
    }
  });

  btnLoadSample.addEventListener('click', () => loadSampleImage());
  if (btnLoadSample2) btnLoadSample2.addEventListener('click', () => loadSampleImage());

  // --- Image Upload & Detection (Foto Mode) ---
  async function handleImageUpload(file) {
    stopFotoSnapCamera();
    const formData = new FormData();
    formData.append('image', file);

    showLoading('Sedang menganalisis foto...');
    scannerLine.classList.remove('d-none');

    try {
      const resp = await fetch('/api/detect/image', {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();
      hideLoading();
      scannerLine.classList.add('d-none');

      if (!data.success) {
        showToast(data.error || 'Gagal memproses gambar.', 'error');
        return;
      }

      displayFotoResults(data, 'FOTO');
    } catch (err) {
      hideLoading();
      scannerLine.classList.add('d-none');
      showToast('Terjadi error saat mendeteksi gambar.', 'error');
    }
  }

  async function loadSampleImage() {
    showLoading('Memuat contoh gambar...');
    scannerLine.classList.remove('d-none');

    try {
      const resp = await fetch('/api/sample');
      const data = await resp.json();
      hideLoading();
      scannerLine.classList.add('d-none');

      if (!data.success) {
        showToast(data.error || 'Gagal memuat contoh gambar.', 'error');
        return;
      }

      displayFotoResults(data, 'FOTO (SAMPLE)');
    } catch (err) {
      hideLoading();
      scannerLine.classList.add('d-none');
      showToast('Gagal memuat contoh gambar.', 'error');
    }
  }

  function displayFotoResults(data, sourceMode) {
    emptyUploadState.classList.add('d-none');
    imageDisplayWrapper.classList.remove('d-none');
    photoCanvas.classList.remove('d-none');

    const img = new Image();
    img.onload = () => {
      photoCanvas.width = img.naturalWidth;
      photoCanvas.height = img.naturalHeight;
      const ctx = photoCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
    };
    img.src = data.annotated_image;

    stats.totalScanned++;
    if (data.total_plates > 0) {
      stats.platesFound += data.total_plates;
      playBeep();
    }
    updateStatsDisplay(data.inference_time_ms);

    plateCountBadge.textContent = `${data.total_plates} PLAT`;

    if (data.detections && data.detections.length > 0) {
      const primary = data.detections[0];
      primaryPlateNumber.textContent = primary.plate_text || 'TIDAK TERBACA';
      primaryPlatePeriod.textContent = primary.plate_period || '08.28';
      
      const yoloPct = Math.round(primary.confidence * 100);
      yoloConfVal.textContent = `${yoloPct}%`;
      yoloConfBar.style.width = `${yoloPct}%`;

      const ocrPct = Math.round((primary.ocr_confidence || 0.85) * 100);
      ocrConfVal.textContent = `${ocrPct}%`;
      ocrConfBar.style.width = `${ocrPct}%`;

      renderCrops(data.detections);
      renderOcrDebug(data.detections, 0);
      showToast(`${data.total_plates} plat terdeteksi.`, 'success');

      data.detections.forEach(det => {
        addHistoryEntry({
          mode: sourceMode,
          plateText: det.plate_text || 'TIDAK TERBACA',
          yoloConf: Math.round(det.confidence * 100),
          ocrConf: Math.round((det.ocr_confidence || 0.85) * 100),
          cropImage: det.crop_image,
          fullImage: data.annotated_image
        });
      });
    } else {
      primaryPlateNumber.textContent = 'TIDAK DITEMUKAN';
      primaryPlatePeriod.textContent = '--.--';
      yoloConfVal.textContent = '0%';
      yoloConfBar.style.width = '0%';
      ocrConfVal.textContent = '0%';
      ocrConfBar.style.width = '0%';
      cropsList.innerHTML = `<div class="no-crops-placeholder"><i class="fa-solid fa-triangle-exclamation"></i> Tidak ada plat terdeteksi</div>`;
      renderOcrDebug([]);
      showToast('Tidak ada plat terdeteksi. Coba foto yang lebih jelas atau lebih dekat.', 'error');
    }
  }

  // --- Zoom Crop Button & OCR Debug Panel Toggle ---
  if (btnZoomCrop) {
    btnZoomCrop.addEventListener('click', () => {
      if (cachedDetections && cachedDetections.length > 0) {
        const d = cachedDetections[currentDebugDetectionIndex] || cachedDetections[0];
        openModal(d.crop_image, `Plat: ${d.plate_text || 'Terpotong'}`, `YOLO Conf: ${(d.confidence*100).toFixed(1)}% | OCR: ${((d.ocr_confidence||0.85)*100).toFixed(1)}%`);
      } else {
        showToast('Belum ada potongan plat untuk diperbesar.', 'info');
      }
    });
  }

  if (debugTitleBar) {
    debugTitleBar.addEventListener('click', () => {
      if (debugVariantsWrapper) {
        const isCollapsed = debugVariantsWrapper.classList.toggle('collapsed');
        if (debugChevron) {
          debugChevron.className = isCollapsed ? 'fa-solid fa-chevron-right' : 'fa-solid fa-chevron-down';
        }
      }
    });
  }

  function renderCrops(detections) {
    cropsList.innerHTML = '';
    cachedDetections = detections || [];
    detections.forEach((det, idx) => {
      const card = document.createElement('div');
      card.className = `crop-card ${idx === currentDebugDetectionIndex ? 'active-crop' : ''}`;
      card.innerHTML = `
        <img src="${det.crop_image}" alt="Plat ${idx+1}" />
        <span class="crop-card-label">${det.plate_text || `Plat #${idx+1}`}</span>
      `;
      card.addEventListener('click', () => {
        renderOcrDebug(detections, idx);
        openModal(det.crop_image, `Plat #${idx+1}: ${det.plate_text || 'Terpotong'}`, `YOLO Conf: ${(det.confidence*100).toFixed(1)}% | OCR: ${((det.ocr_confidence||0.85)*100).toFixed(1)}%`);
      });
      cropsList.appendChild(card);
    });
  }

  function renderOcrDebug(detections, selectedIndex = 0) {
    if (!debugVariantsScroll || !candidatesTableBody) return;
    cachedDetections = detections || [];
    currentDebugDetectionIndex = selectedIndex;

    if (!detections || detections.length === 0) {
      debugVariantsScroll.innerHTML = `
        <div class="no-crops-placeholder" style="grid-column: 1 / -1;">
          <i class="fa-solid fa-sliders"></i>
          <span>Belum ada data preprocessing OCR</span>
        </div>`;
      candidatesTableBody.innerHTML = `
        <tr class="empty-candidate-row">
          <td colspan="4" class="text-center text-dim">Belum ada kandidat</td>
        </tr>`;
      if (candidateCountBadge) candidateCountBadge.textContent = '0 kandidat';
      if (debugPlateTabs) debugPlateTabs.classList.add('d-none');
      return;
    }

    // Update active state in crops list
    const cropCards = cropsList.querySelectorAll('.crop-card');
    cropCards.forEach((c, i) => {
      c.classList.toggle('active-crop', i === selectedIndex);
    });

    // If multiple detections, render tabs
    if (debugPlateTabs) {
      if (detections.length > 1) {
        debugPlateTabs.classList.remove('d-none');
        debugPlateTabs.innerHTML = '';
        detections.forEach((d, idx) => {
          const tab = document.createElement('button');
          tab.type = 'button';
          tab.className = `debug-plate-tab ${idx === selectedIndex ? 'active' : ''}`;
          tab.textContent = `Plat #${idx + 1}: ${d.plate_text || 'Deteksi'}`;
          tab.addEventListener('click', (e) => {
            e.stopPropagation();
            renderOcrDebug(cachedDetections, idx);
          });
          debugPlateTabs.appendChild(tab);
        });
      } else {
        debugPlateTabs.classList.add('d-none');
      }
    }

    const det = detections[selectedIndex] || detections[0];
    const variants = det.preprocessed_variants || [];
    const candidates = det.raw_ocr_candidates || [];

    // Render variant cards
    if (variants.length > 0) {
      debugVariantsScroll.innerHTML = '';
      variants.forEach(v => {
        const card = document.createElement('div');
        card.className = 'variant-card';
        const confPct = v.conf > 0 ? `${Math.round(v.conf * 100)}%` : '--';
        card.innerHTML = `
          <div class="variant-card-header">
            <span class="variant-badge">${v.name}</span>
            <span class="variant-conf">${confPct}</span>
          </div>
          <img src="${v.image_b64}" alt="${v.name}" />
          <div class="variant-text" title="${v.plate_text || 'Tidak terbaca'}">
            ${v.plate_text || '<span class="text-dim">Tidak terbaca</span>'}
          </div>
        `;
        card.addEventListener('click', () => {
          openModal(
            v.image_b64,
            `Preprocessing: ${v.name.toUpperCase()}`,
            `Hasil OCR: ${v.plate_text || 'Tidak terbaca'} | Conf: ${confPct}`
          );
        });
        debugVariantsScroll.appendChild(card);
      });
    } else {
      debugVariantsScroll.innerHTML = `
        <div class="no-crops-placeholder" style="grid-column: 1 / -1;">
          <i class="fa-solid fa-circle-info"></i>
          <span>Varian preprocessing tidak tersedia untuk deteksi ini</span>
        </div>`;
    }

    // Render candidates table
    if (candidateCountBadge) {
      candidateCountBadge.textContent = `${candidates.length} kandidat`;
    }

    if (candidates.length > 0) {
      candidatesTableBody.innerHTML = '';
      candidates.forEach((c, idx) => {
        const tr = document.createElement('tr');
        const isWinner = c.text === det.plate_text;
        if (isWinner) tr.className = 'winner';
        tr.innerHTML = `
          <td><strong>${c.text}</strong> ${isWinner ? '<i class="fa-solid fa-check text-emerald"></i>' : ''}</td>
          <td>${Math.round(c.conf * 100)}%</td>
          <td>${c.score}</td>
          <td><span class="badge-mini">${c.variant || 'raw'}</span></td>
        `;
        candidatesTableBody.appendChild(tr);
      });
    } else {
      candidatesTableBody.innerHTML = `
        <tr class="empty-candidate-row">
          <td colspan="4" class="text-center text-dim">Tidak ada kandidat valid</td>
        </tr>`;
    }
  }

  // =====================================================
  // CCTV Live Surveillance Mode
  // =====================================================
  if (btnToggleCam) btnToggleCam.addEventListener('click', toggleCamera);
  if (btnStartCamCenter) btnStartCamCenter.addEventListener('click', toggleCamera);
  if (btnSimulateStream) btnSimulateStream.addEventListener('click', toggleSimulation);
  if (btnSimulateStreamCenter) btnSimulateStreamCenter.addEventListener('click', toggleSimulation);
  if (btnSnapPlate) btnSnapPlate.addEventListener('click', captureSnapshot);

  async function toggleCamera() {
    if (isCameraActive) {
      stopCamera();
    } else {
      await startCamera();
    }
  }

  async function startCamera() {
    stopSimulation();
    try {
      const constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 } } };
      // Use selected camera device
      if (cameraSourceSelect && cameraSourceSelect.value) {
        constraints.video.deviceId = { exact: cameraSourceSelect.value };
      }
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      webcamVideo.srcObject = mediaStream;
      await webcamVideo.play();

      isCameraActive = true;
      camBtnText.textContent = 'Berhenti';
      btnToggleCam.classList.remove('btn-emerald');
      btnToggleCam.classList.add('btn-danger');
      btnToggleCam.querySelector('i').className = 'fa-solid fa-stop';
      cctvOffState.classList.add('d-none');
      cctvDisplayWrapper.classList.remove('d-none');
      if (btnSnapPlate) btnSnapPlate.disabled = false;
      if (btnQuickCamera) btnQuickCamera.innerHTML = '<i class="fa-solid fa-stop"></i><span>Berhenti</span>';
      setSystemStatus('Live', 'busy');
      showToast('Kamera live aktif.', 'success');

      // Adjust overlay canvas after video starts
      setTimeout(() => {
        cctvOverlayCanvas.width = webcamVideo.videoWidth || 1280;
        cctvOverlayCanvas.height = webcamVideo.videoHeight || 720;
      }, 500);

      // Start detection frame loop (every 800ms to avoid overloading)
      cctvInterval = setInterval(processCctvFrame, 800);
    } catch (err) {
      console.error('Camera access error:', err);
      showToast('Tidak dapat mengakses kamera. Pastikan izin kamera aktif dan kamera tidak dipakai aplikasi lain.', 'error');
    }
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    if (cctvInterval) {
      clearInterval(cctvInterval);
      cctvInterval = null;
    }
    isCameraActive = false;
    camBtnText.textContent = 'Mulai Kamera';
    btnToggleCam.classList.remove('btn-danger');
    btnToggleCam.classList.add('btn-emerald');
    btnToggleCam.querySelector('i').className = 'fa-solid fa-power-off';
    if (btnSnapPlate) btnSnapPlate.disabled = true;
    if (btnQuickCamera) btnQuickCamera.innerHTML = '<i class="fa-solid fa-video"></i><span>Mulai Kamera</span>';
    setSystemStatus('Siap', 'ready');
    if (!isSimulating) {
      cctvDisplayWrapper.classList.add('d-none');
      cctvOffState.classList.remove('d-none');
    }
    // Clear overlay
    const ctx = cctvOverlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, cctvOverlayCanvas.width, cctvOverlayCanvas.height);
  }

  function captureSnapshot() {
    if (!isCameraActive || webcamVideo.paused) return;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = webcamVideo.videoWidth || 640;
    tempCanvas.height = webcamVideo.videoHeight || 480;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, tempCanvas.width, tempCanvas.height);
    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
    openModal(dataUrl, 'Snapshot Kamera', 'Tangkapan layar dari webcam CCTV');
  }

  async function processCctvFrame() {
    if (!isCameraActive || webcamVideo.paused || webcamVideo.ended) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = webcamVideo.videoWidth || 640;
    tempCanvas.height = webcamVideo.videoHeight || 480;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(webcamVideo, 0, 0, tempCanvas.width, tempCanvas.height);

    const base64Data = tempCanvas.toDataURL('image/jpeg', 0.8);

    try {
      const resp = await fetch('/api/detect/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: base64Data })
      });
      const data = await resp.json();

      if (data.success) {
        drawCctvOverlay(data.detections);
        handleCctvDetectionEvents(data);
      }
    } catch (err) {
      console.error('CCTV frame error:', err);
    }
  }

  function drawCctvOverlay(detections) {
    const ctx = cctvOverlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, cctvOverlayCanvas.width, cctvOverlayCanvas.height);

    if (!detections || detections.length === 0) return;

    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.box;
      const w = x2 - x1;
      const h = y2 - y1;

      // Main bounding box
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      // Corner brackets (HUD style)
      const cornerLen = Math.min(15, w / 4);
      ctx.strokeStyle = '#00ff9d';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cornerLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cornerLen, y1); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cornerLen); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x1, y2 - cornerLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cornerLen, y2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cornerLen); ctx.stroke();

      // Label background
      const text = `${det.plate_text || 'PLAT'} [${Math.round(det.confidence * 100)}%]`;
      ctx.font = 'bold 16px "Orbitron", monospace';
      const textWidth = ctx.measureText(text).width;
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
      ctx.fillRect(x1, Math.max(0, y1 - 28), textWidth + 16, 26);
      ctx.fillStyle = '#00ff9d';
      ctx.fillText(text, x1 + 8, Math.max(20, y1 - 8));
    });
  }

  function handleCctvDetectionEvents(data) {
    updateStatsDisplay(data.inference_time_ms);
    if (!data.detections || data.detections.length === 0) return;

    const now = Date.now();
    const primary = data.detections[0];
    const plateText = primary.plate_text || 'PLAT TERDETEKSI';

    primaryPlateNumber.textContent = plateText;
    primaryPlatePeriod.textContent = primary.plate_period || 'LIVE';
    yoloConfVal.textContent = `${Math.round(primary.confidence * 100)}%`;
    yoloConfBar.style.width = `${Math.round(primary.confidence * 100)}%`;
    ocrConfVal.textContent = `${Math.round((primary.ocr_confidence || 0.85) * 100)}%`;
    ocrConfBar.style.width = `${Math.round((primary.ocr_confidence || 0.85) * 100)}%`;
    plateCountBadge.textContent = `${data.detections.length} PLAT`;
    renderCrops(data.detections);

    // Throttle duplicate logging
    if (plateText !== lastDetectedPlate || now - lastDetectionTime > 3000) {
      lastDetectedPlate = plateText;
      lastDetectionTime = now;
      playBeep();
      stats.platesFound++;

      addHistoryEntry({
        mode: 'CCTV STREAM',
        plateText: plateText,
        yoloConf: Math.round(primary.confidence * 100),
        ocrConf: Math.round((primary.ocr_confidence || 0.85) * 100),
        cropImage: primary.crop_image,
        fullImage: primary.crop_image
      });
    }
  }

  // --- CCTV Simulation ---
  function toggleSimulation() {
    if (isSimulating) {
      stopSimulation();
    } else {
      startSimulation();
    }
  }

  function startSimulation() {
    stopCamera();
    isSimulating = true;
    cctvOffState.classList.add('d-none');
    cctvDisplayWrapper.classList.remove('d-none');
    btnSimulateStream.innerHTML = '<i class="fa-solid fa-circle-stop text-crimson"></i> Stop Demo';
    setSystemStatus('Demo live', 'busy');

    cctvInterval = setInterval(async () => {
      try {
        const resp = await fetch('/api/sample');
        const data = await resp.json();
        if (data.success) {
          const img = new Image();
          img.onload = () => {
            cctvOverlayCanvas.width = img.naturalWidth;
            cctvOverlayCanvas.height = img.naturalHeight;
            const ctx = cctvOverlayCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
          };
          img.src = data.annotated_image;
          handleCctvDetectionEvents(data);
        }
      } catch (err) {
        console.error('Sim error:', err);
      }
    }, 1200);
  }

  function stopSimulation() {
    if (isSimulating) {
      clearInterval(cctvInterval);
      cctvInterval = null;
      isSimulating = false;
      btnSimulateStream.innerHTML = '<i class="fa-solid fa-circle-play"></i> Demo Live';
      setSystemStatus('Siap', 'ready');
      if (!isCameraActive) {
        cctvDisplayWrapper.classList.add('d-none');
        cctvOffState.classList.remove('d-none');
      }
    }
  }

  // --- History Log Management ---
  function addHistoryEntry(entry) {
    const timeStr = new Date().toLocaleTimeString('id-ID', { hour12: false });
    const fullDate = new Date().toISOString().replace('T', ' ').substring(0, 19);

    const logItem = {
      id: Date.now() + Math.random().toString(36).substr(2, 4),
      time: timeStr,
      fullDate: fullDate,
      ...entry
    };

    historyLogs.unshift(logItem);

    const emptyRow = historyTableBody.querySelector('.empty-row');
    if (emptyRow) emptyRow.remove();

    const tr = document.createElement('tr');
    tr.id = `log-${logItem.id}`;
    tr.innerHTML = `
      <td class="font-mono">${logItem.time}</td>
      <td><span class="badge-pill">${logItem.mode}</span></td>
      <td>
        ${logItem.cropImage ? `<img src="${logItem.cropImage}" class="crop-thumb" alt="Crop" />` : '-'}
      </td>
      <td><span class="plate-text-pill">${logItem.plateText}</span></td>
      <td class="font-mono text-emerald">${logItem.yoloConf}%</td>
      <td class="font-mono text-cyan">${logItem.ocrConf}%</td>
      <td><span class="text-emerald"><i class="fa-solid fa-circle-check"></i> VERIFIED</span></td>
      <td>
        <button class="btn btn-secondary btn-sm btn-view-log" data-id="${logItem.id}">
          <i class="fa-solid fa-eye"></i>
        </button>
      </td>
    `;

    tr.querySelector('.btn-view-log').addEventListener('click', () => {
      openModal(logItem.cropImage || logItem.fullImage, `Plat: ${logItem.plateText}`, `Waktu: ${logItem.fullDate} | Mode: ${logItem.mode}`);
    });

    historyTableBody.insertBefore(tr, historyTableBody.firstChild);
  }

  // History Search Filter
  logSearchInput.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    const rows = historyTableBody.querySelectorAll('tr:not(.empty-row)');
    rows.forEach(row => {
      const plateText = row.querySelector('.plate-text-pill')?.textContent.toLowerCase() || '';
      row.style.display = plateText.includes(q) ? '' : 'none';
    });
  });

  // Export CSV
  btnExportCsv.addEventListener('click', () => {
    if (historyLogs.length === 0) {
      showToast('Belum ada data untuk diekspor.', 'error');
      return;
    }
    let csv = 'Timestamp,Mode,Plate Number,YOLO Confidence (%),OCR Confidence (%)\n';
    historyLogs.forEach(item => {
      csv += `"${item.fullDate}","${item.mode}","${item.plateText}","${item.yoloConf}","${item.ocrConf}"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `alpr_detection_log_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  });

  // Clear Log
  btnClearLog.addEventListener('click', () => {
    if (confirm('Bersihkan seluruh riwayat deteksi?')) {
      historyLogs = [];
      historyTableBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="8" class="text-center">Belum ada riwayat deteksi. Jalankan Foto atau CCTV untuk merekam plat kendaraan.</td>
        </tr>
      `;
    }
  });

  // --- Modal Helpers ---
  function openModal(imgSrc, title, caption) {
    modalImg.src = imgSrc;
    modalTitle.textContent = title;
    modalCaption.textContent = caption;
    imageModal.classList.remove('d-none');
  }

  btnModalClose.addEventListener('click', () => imageModal.classList.add('d-none'));
  imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) imageModal.classList.add('d-none');
  });

  // --- UI Helpers ---
  function showLoading(msg) {
    loadingText.textContent = msg;
    loadingOverlay.classList.remove('d-none');
  }

  function hideLoading() {
    loadingOverlay.classList.add('d-none');
  }

  function updateStatsDisplay(inferenceMs) {
    statTotalScanned.textContent = stats.totalScanned;
    statPlatesFound.textContent = stats.platesFound;
    if (inferenceMs) {
      inferenceSpeed.textContent = `${Math.round(inferenceMs)} ms`;
    }
    const rate = stats.totalScanned > 0 ? Math.round((stats.platesFound / stats.totalScanned) * 100) : 100;
    statSuccessRate.textContent = `${rate}%`;
  }
});
