/**
 * STAGE 8B: LIVE DASHBOARD & INCIDENT COMMAND CENTER CONTROLLER
 * Wearable Fall Detection System
 * Pure Vanilla JavaScript — High-Performance, Accessible, Resilient
 */

(function () {
  "use strict";

  // --------------------------------------------------------------------------
  // Configuration & Constants
  // --------------------------------------------------------------------------
  const CONFIG = {
    POLLING_INTERVAL_MS: 1000,        // Backward-compatible polling interval (Stage 5B/6H)
    HEALTH_POLL_INTERVAL_MS: 3000,    // Poll /api/system/health every 3.0s (Section 3 & 12)
    EVENT_POLL_INTERVAL_MS: 1000,     // Poll /api/fall-events every 1.0s (Section 12: Fast incident discovery)
    DEFAULT_RESPONSE_WINDOW_SEC: 60,  // Incident countdown response window in seconds
    PAGE_SIZE: 15,                    // History log records per page
    DEFAULT_COORDINATES: [22.728421, 75.869163] // Default map view center (Indore, India)
  };

  // --------------------------------------------------------------------------
  // Application State
  // --------------------------------------------------------------------------
  const STATE = {
    sessionStartTimeMs: Date.now(),   // Dashboard session start timestamp boundary
    activeIncident: null,             // Currently active incident (CONFIRMED or ACKNOWLEDGED)
    selectedIncident: null,           // Incident currently selected in history / focused on map
    systemHealth: null,               // Latest payload from GET /api/system/health
    countdownRemainingSec: 0,         // Remaining seconds in response window
    countdownIntervalId: null,        // Timer interval ID for active response countdown
    healthIntervalId: null,           // Interval ID for system health polling
    eventIntervalId: null,            // Interval ID for fall event polling
    currentPage: 1,                   // Current pagination page in history log
    totalEventsCount: 0,              // Total event count reported by backend
    isBackendOnline: true,            // Tracks backend API connectivity
    inFlightAction: false,            // Mutex preventing duplicate user actions
    cancelledEventIds: new Set(),     // Set of event_ids cancelled in current session
    resolvedEventIds: new Set(),      // Set of event_ids resolved in current session
    map: null,                        // Leaflet map instance
    currentMarker: null               // Leaflet marker for selected/active incident
  };

  // --------------------------------------------------------------------------
  // DOM References
  // --------------------------------------------------------------------------
  const DOM = {
    // Header & System Status (Section 3)
    systemStatusBox: document.getElementById("systemStatusBox"),
    systemStatusDot: document.getElementById("systemStatusDot"),
    systemStatusText: document.getElementById("systemStatusText"),

    // Subsystem Health Indicators (Section 3 & 10)
    pillEsp32: document.getElementById("pillEsp32"),
    dotEsp32: document.getElementById("dotEsp32"),
    valEsp32: document.getElementById("valEsp32"),

    pillMqtt: document.getElementById("pillMqtt"),
    dotMqtt: document.getElementById("dotMqtt"),
    valMqtt: document.getElementById("valMqtt"),

    pillGateway: document.getElementById("pillGateway"),
    dotGateway: document.getElementById("dotGateway"),
    valGateway: document.getElementById("valGateway"),

    pillBackend: document.getElementById("pillBackend"),
    dotBackend: document.getElementById("dotBackend"),
    valBackend: document.getElementById("valBackend"),

    pillDatabase: document.getElementById("pillDatabase"),
    dotDatabase: document.getElementById("dotDatabase"),
    valDatabase: document.getElementById("valDatabase"),

    pillGps: document.getElementById("pillGps"),
    dotGps: document.getElementById("dotGps"),
    valGps: document.getElementById("valGps"),

    // Connection Alert Banner (Section 11)
    serviceAlertBanner: document.getElementById("serviceAlertBanner"),
    serviceAlertMessage: document.getElementById("serviceAlertMessage"),
    btnRetryConnection: document.getElementById("btnRetryConnection"),

    // Normal Monitoring State (Section 4)
    noActiveIncidentSection: document.getElementById("noActiveIncidentSection"),

    // Active Fall Incident Panel (Section 4, 5, 6)
    activeAlertSection: document.getElementById("activeAlertSection"),
    alertSeverityBadge: document.getElementById("alertSeverityBadge"),
    alertHeading: document.getElementById("alertHeading"),
    alertStatusBadge: document.getElementById("alertStatusBadge"),
    alertStatusText: document.getElementById("alertStatusText"),
    alertCountdownText: document.getElementById("alertCountdownText"),
    countdownProgressBar: document.getElementById("countdownProgressBar"),
    alertEventId: document.getElementById("alertEventId"),
    alertDetectedTime: document.getElementById("alertDetectedTime"),
    alertConfidence: document.getElementById("alertConfidence"),
    alertSource: document.getElementById("alertSource"),
    alertSessionId: document.getElementById("alertSessionId"),

    // Incident Location Elements (Section 7)
    alertLocationSection: document.getElementById("alertLocationSection"),
    alertGpsStatusBadge: document.getElementById("alertGpsStatusBadge"),
    alertLatitude: document.getElementById("alertLatitude"),
    alertLongitude: document.getElementById("alertLongitude"),
    alertSatellites: document.getElementById("alertSatellites"),
    alertHdop: document.getElementById("alertHdop"),
    alertMapLink: document.getElementById("alertMapLink"),

    // Action Buttons (Section 5 & 6)
    btnAcknowledge: document.getElementById("btnAcknowledge"),
    btnResolve: document.getElementById("btnResolve"),
    btnOpenCancelDialog: document.getElementById("btnOpenCancelDialog"),
    cancelReasonPanel: document.getElementById("cancelReasonPanel"),
    cancelReasonInput: document.getElementById("cancelReasonInput"),
    btnConfirmCancel: document.getElementById("btnConfirmCancel"),
    btnDismissCancel: document.getElementById("btnDismissCancel"),
    alertActionFeedback: document.getElementById("alertActionFeedback"),

    // Status / Telemetry Card
    telemetryStatusBadge: document.getElementById("telemetryStatusBadge"),
    noActiveEventPlaceholder: document.getElementById("noActiveEventPlaceholder"),
    activeEventDetailsView: document.getElementById("activeEventDetailsView"),
    viewEventId: document.getElementById("viewEventId"),
    viewStatusBadge: document.getElementById("viewStatusBadge"),
    viewDetectedTime: document.getElementById("viewDetectedTime"),
    viewOccurredTime: document.getElementById("viewOccurredTime"),
    viewConfidence: document.getElementById("viewConfidence"),
    viewSource: document.getElementById("viewSource"),
    viewCountdownStatus: document.getElementById("viewCountdownStatus"),
    viewGpsStatusBadge: document.getElementById("viewGpsStatusBadge"),
    viewLatitude: document.getElementById("viewLatitude"),
    viewLongitude: document.getElementById("viewLongitude"),
    viewSatellitesHdop: document.getElementById("viewSatellitesHdop"),
    viewMapRow: document.getElementById("viewMapRow"),
    viewMapLink: document.getElementById("viewMapLink"),
    btnPanelAck: document.getElementById("btnPanelAck"),
    btnPanelResolve: document.getElementById("btnPanelResolve"),
    btnPanelCancel: document.getElementById("btnPanelCancel"),

    // Map Card (Section 7)
    incidentMap: document.getElementById("incidentMap"),
    mapFallbackContainer: document.getElementById("mapFallbackContainer"),
    mapFallbackText: document.getElementById("mapFallbackText"),
    mapStatusBadge: document.getElementById("mapStatusBadge"),
    mapUnavailableOverlay: document.getElementById("mapUnavailableOverlay"),
    mapLastKnownBadge: document.getElementById("mapLastKnownBadge"),
    mapCoordSummary: document.getElementById("mapCoordSummary"),
    mapExternalLink: document.getElementById("mapExternalLink"),

    // History Table (Section 8)
    historyTable: document.getElementById("historyTable"),
    historyTableBody: document.getElementById("historyTableBody"),
    historyLoading: document.getElementById("historyLoading"),
    historyEmpty: document.getElementById("historyEmpty"),
    historyError: document.getElementById("historyError"),
    historyErrorMessage: document.getElementById("historyErrorMessage"),
    btnRefreshHistory: document.getElementById("btnRefreshHistory"),
    btnPrevPage: document.getElementById("btnPrevPage"),
    btnNextPage: document.getElementById("btnNextPage"),
    paginationInfo: document.getElementById("paginationInfo"),

    // Modal (Section 9)
    eventDetailModal: document.getElementById("eventDetailModal"),
    modalContent: document.getElementById("modalContent"),
    btnCloseModal: document.getElementById("btnCloseModal"),
    btnModalClose: document.getElementById("btnModalClose")
  };

  // --------------------------------------------------------------------------
  // Utility Functions
  // --------------------------------------------------------------------------
  function formatTimestamp(isoString) {
    if (!isoString) return "--";
    try {
      const d = new Date(isoString);
      if (isNaN(d.getTime())) return isoString;
      return d.toLocaleTimeString([], { hour12: false }) + " (" + d.toISOString().slice(0, 10) + ")";
    } catch {
      return isoString;
    }
  }

  function formatConfidence(conf) {
    if (conf === null || conf === undefined) return "--";
    const num = Number(conf);
    if (isNaN(num)) return "--";
    return (num <= 1.0 ? (num * 100).toFixed(1) : num.toFixed(1)) + "%";
  }

  function renderStatusBadge(status) {
    const safeStatus = (status || "UNKNOWN").toUpperCase();
    return `<span class="badge badge-${safeStatus}">${safeStatus}</span>`;
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // --------------------------------------------------------------------------
  // Network Transport & Resilient Fetch
  // --------------------------------------------------------------------------
  async function apiFetch(endpoint, options = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
      const response = await fetch(endpoint, {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
          ...(options.headers || {})
        }
      });
      clearTimeout(timeoutId);

      if (!STATE.isBackendOnline) {
        handleBackendConnectionRestored();
      }
      return response;
    } catch (err) {
      clearTimeout(timeoutId);
      handleBackendConnectionLost();
      throw err;
    }
  }

  function handleBackendConnectionLost() {
    if (!STATE.isBackendOnline) return;
    STATE.isBackendOnline = false;

    // Show non-intrusive alert banner (Section 11)
    if (DOM.serviceAlertBanner) {
      DOM.serviceAlertBanner.className = "service-alert-banner";
      DOM.serviceAlertMessage.textContent = "BACKEND CONNECTION LOST — Attempting reconnection...";
      DOM.serviceAlertBanner.classList.remove("hidden");
    }

    // Update system status indicator
    if (DOM.systemStatusBox) {
      DOM.systemStatusBox.className = "system-status-indicator-box status-error";
      DOM.systemStatusDot.className = "system-status-dot dot-error";
      DOM.systemStatusText.textContent = "BACKEND OFFLINE";
    }
  }

  function handleBackendConnectionRestored() {
    STATE.isBackendOnline = true;

    if (DOM.serviceAlertBanner) {
      DOM.serviceAlertMessage.textContent = "SYSTEM CONNECTION RESTORED";
      setTimeout(() => {
        if (STATE.isBackendOnline && DOM.serviceAlertBanner) {
          DOM.serviceAlertBanner.classList.add("hidden");
        }
      }, 2500);
    }
  }

  // --------------------------------------------------------------------------
  // Section 3 & 10: System Health Monitoring & Degradation Handling
  // --------------------------------------------------------------------------
  async function pollSystemHealth() {
    try {
      const res = await apiFetch("/api/system/health");
      if (!res.ok) return;

      const health = await res.json();
      STATE.systemHealth = health;
      renderSystemHealth(health);
    } catch (err) {
      // Offline banner handled by apiFetch
    }
  }

  function renderSystemHealth(h) {
    if (!h) return;

    // Overall Status
    const overall = (h.status || "SYSTEM_READY").toUpperCase();
    if (DOM.systemStatusBox) {
      if (overall === "SYSTEM_READY") {
        DOM.systemStatusBox.className = "system-status-indicator-box";
        DOM.systemStatusDot.className = "system-status-dot dot-ready";
        DOM.systemStatusText.textContent = "SYSTEM READY";
      } else if (overall === "SYSTEM_DEGRADED") {
        DOM.systemStatusBox.className = "system-status-indicator-box status-degraded";
        DOM.systemStatusDot.className = "system-status-dot dot-degraded";
        DOM.systemStatusText.textContent = "SYSTEM DEGRADED";
      } else if (overall === "SYSTEM_ERROR") {
        DOM.systemStatusBox.className = "system-status-indicator-box status-error";
        DOM.systemStatusDot.className = "system-status-dot dot-error";
        DOM.systemStatusText.textContent = "SYSTEM ERROR";
      } else if (overall === "SYSTEM_STARTING") {
        DOM.systemStatusBox.className = "system-status-indicator-box status-starting";
        DOM.systemStatusDot.className = "system-status-dot dot-starting";
        DOM.systemStatusText.textContent = "SYSTEM STARTING";
      }
    }

    // Subsystem: ESP32
    if (h.esp32) {
      const espStatus = (h.esp32.status || "").toUpperCase();
      if (espStatus === "CONNECTED") {
        setSubsystemPill(DOM.pillEsp32, DOM.dotEsp32, DOM.valEsp32, "Connected", "online");
      } else if (espStatus === "STALE") {
        setSubsystemPill(DOM.pillEsp32, DOM.dotEsp32, DOM.valEsp32, "Stale", "warning");
      } else {
        setSubsystemPill(DOM.pillEsp32, DOM.dotEsp32, DOM.valEsp32, "Disconnected", "offline");
      }
    }

    // Subsystem: MQTT
    if (h.mqtt) {
      const mqttStatus = (h.mqtt.status || "").toUpperCase();
      if (mqttStatus === "CONNECTED") {
        setSubsystemPill(DOM.pillMqtt, DOM.dotMqtt, DOM.valMqtt, "Connected", "online");
      } else {
        setSubsystemPill(DOM.pillMqtt, DOM.dotMqtt, DOM.valMqtt, "Disconnected", "offline");
      }
    }

    // Subsystem: Gateway
    if (h.gateway) {
      const gwStatus = (h.gateway.status || "").toUpperCase();
      if (gwStatus === "READY") {
        setSubsystemPill(DOM.pillGateway, DOM.dotGateway, DOM.valGateway, "Ready", "online");
      } else {
        setSubsystemPill(DOM.pillGateway, DOM.dotGateway, DOM.valGateway, "Stopped", "warning");
      }
    }

    // Subsystem: Backend
    if (h.backend) {
      const backendStatus = (h.backend.status || "").toUpperCase();
      if (backendStatus === "READY") {
        setSubsystemPill(DOM.pillBackend, DOM.dotBackend, DOM.valBackend, "Ready", "online");
      } else {
        setSubsystemPill(DOM.pillBackend, DOM.dotBackend, DOM.valBackend, "Error", "offline");
      }
    }

    // Subsystem: Database
    if (h.database) {
      const dbStatus = (h.database.status || "").toUpperCase();
      if (dbStatus === "READY") {
        setSubsystemPill(DOM.pillDatabase, DOM.dotDatabase, DOM.valDatabase, "Ready", "online");
      } else {
        setSubsystemPill(DOM.pillDatabase, DOM.dotDatabase, DOM.valDatabase, "Error", "offline");
      }
    }

    // Subsystem: GPS
    if (h.gps) {
      const gpsStatus = (h.gps.status || "").toUpperCase();
      if (gpsStatus === "AVAILABLE" && h.gps.fix_valid) {
        setSubsystemPill(DOM.pillGps, DOM.dotGps, DOM.valGps, "Available", "online");
      } else if (gpsStatus === "AVAILABLE") {
        setSubsystemPill(DOM.pillGps, DOM.dotGps, DOM.valGps, "No 3D Fix", "warning");
      } else {
        setSubsystemPill(DOM.pillGps, DOM.dotGps, DOM.valGps, "Unavailable", "warning");
      }
    }
  }

  function setSubsystemPill(pillEl, dotEl, valEl, text, state) {
    if (!pillEl || !dotEl || !valEl) return;
    valEl.textContent = text;

    pillEl.classList.remove("pill-degraded", "pill-error");
    dotEl.classList.remove("dot-online", "dot-warning", "dot-offline");

    if (state === "online") {
      dotEl.classList.add("dot-online");
    } else if (state === "warning") {
      pillEl.classList.add("pill-degraded");
      dotEl.classList.add("dot-warning");
    } else {
      pillEl.classList.add("pill-error");
      dotEl.classList.add("dot-offline");
    }
  }

  // --------------------------------------------------------------------------
  // Section 4, 5, 6: Fall Event Polling & Incident Management
  // --------------------------------------------------------------------------
  async function pollEvents() {
    if (STATE.inFlightAction) return;
    try {
      const offset = (STATE.currentPage - 1) * CONFIG.PAGE_SIZE;
      const res = await apiFetch(`/api/fall-events?limit=${CONFIG.PAGE_SIZE}&offset=${offset}`);
      if (!res.ok) return;

      const data = await res.json();
      const events = data.events || [];
      STATE.totalEventsCount = data.count || events.length;

      renderHistory(events);
      if (!STATE.inFlightAction) {
        syncActiveIncident(events);
      }
    } catch (err) {
      // Offline handled by apiFetch
    }
  }

  function isEventActiveConfirmed(event) {
    if (!event || event.status !== "CONFIRMED") return false;
    if (STATE.cancelledEventIds && STATE.cancelledEventIds.has(event.event_id)) return false;
    if (STATE.resolvedEventIds && STATE.resolvedEventIds.has(event.event_id)) return false;
    const detectMs = new Date(event.detected_at).getTime();
    if (isNaN(detectMs)) return false;
    const nowMs = Date.now();
    const elapsedSec = (nowMs - detectMs) / 1000;

    // Condition 1: detected_at is within DEFAULT_RESPONSE_WINDOW_SEC (with 5s clock skew tolerance)
    const isWithinWindow = elapsedSec >= -5 && elapsedSec <= CONFIG.DEFAULT_RESPONSE_WINDOW_SEC;

    // Condition 2: demonstrably newer than current dashboard monitoring session/page-load boundary
    const isNewerThanSession = detectMs >= (STATE.sessionStartTimeMs - 5000);

    return isWithinWindow || isNewerThanSession;
  }

  function syncActiveIncident(events) {
    if (STATE.inFlightAction) return;

    // Priority 1: Unacknowledged active confirmed emergency
    // Filter out old historical test events; prefer newest valid active CONFIRMED by detected_at
    const activeConfirmedCandidates = events.filter(isEventActiveConfirmed);
    activeConfirmedCandidates.sort((a, b) => {
      const tA = new Date(a.detected_at).getTime() || 0;
      const tB = new Date(b.detected_at).getTime() || 0;
      return tB - tA;
    });
    const confirmed = activeConfirmedCandidates.length > 0 ? activeConfirmedCandidates[0] : null;

    // Priority 2: Acknowledged incident awaiting resolution
    let acknowledged = null;
    if (STATE.activeIncident && STATE.activeIncident.status === "ACKNOWLEDGED") {
      acknowledged = events.find(e => e.event_id === STATE.activeIncident.event_id && e.status === "ACKNOWLEDGED") || null;
    }
    if (!acknowledged && STATE.activeIncident) {
      acknowledged = events.find(e => {
        if (e.status !== "ACKNOWLEDGED") return false;
        const dMs = new Date(e.detected_at).getTime();
        return !isNaN(dMs) && dMs >= (STATE.sessionStartTimeMs - 5000);
      }) || null;
    }

    const currentTarget = confirmed || acknowledged || null;

    if (currentTarget) {
      setActiveIncident(currentTarget);
    } else {
      clearActiveIncident();
      // If we have history events and no selected incident, display the latest historical location on map
      if (events.length > 0 && !STATE.selectedIncident) {
        showIncidentLocationOnMap(events[0]);
      }
    }
  }

  function setActiveIncident(event) {
    const isSameEvent = STATE.activeIncident && STATE.activeIncident.event_id === event.event_id;
    const isSameStatus = STATE.activeIncident && STATE.activeIncident.status === event.status;

    STATE.activeIncident = event;
    STATE.selectedIncident = event;

    // Show Hero Alert section, hide No-Active-Incident section (Section 4)
    if (DOM.noActiveIncidentSection) DOM.noActiveIncidentSection.classList.add("hidden");
    if (DOM.activeAlertSection) DOM.activeAlertSection.classList.remove("hidden");

    // Populate active incident metadata
    if (DOM.alertEventId) DOM.alertEventId.textContent = event.event_id;
    if (DOM.alertDetectedTime) DOM.alertDetectedTime.textContent = formatTimestamp(event.detected_at);
    if (DOM.alertConfidence) DOM.alertConfidence.textContent = formatConfidence(event.confidence);
    if (DOM.alertSource) DOM.alertSource.textContent = event.source || "ESP32_MPU6050";
    if (DOM.alertSessionId) DOM.alertSessionId.textContent = event.session_id || "--";
    if (DOM.alertStatusText) DOM.alertStatusText.textContent = event.status;

    // Status-specific badges & buttons (Section 5 & 6)
    if (event.status === "CONFIRMED") {
      if (DOM.alertSeverityBadge) {
        DOM.alertSeverityBadge.textContent = "EMERGENCY";
        DOM.alertSeverityBadge.className = "alert-badge-pulse";
      }
      if (DOM.alertHeading) DOM.alertHeading.textContent = "ACTIVE FALL INCIDENT";
      if (DOM.alertStatusBadge) {
        DOM.alertStatusBadge.textContent = "CONFIRMED";
        DOM.alertStatusBadge.className = "badge badge-CONFIRMED";
      }

      // Show Acknowledge, Hide Resolve
      if (DOM.btnAcknowledge) DOM.btnAcknowledge.classList.remove("hidden");
      if (DOM.btnResolve) DOM.btnResolve.classList.add("hidden");
      if (DOM.btnPanelAck) DOM.btnPanelAck.classList.remove("hidden");
      if (DOM.btnPanelResolve) DOM.btnPanelResolve.classList.add("hidden");

      // Start countdown ONLY if not already running for this same event_id
      if (!isSameEvent || !STATE.countdownIntervalId) {
        startCountdown(event);
      }

    } else if (event.status === "ACKNOWLEDGED") {
      if (DOM.alertSeverityBadge) {
        DOM.alertSeverityBadge.textContent = "ACKNOWLEDGED";
        DOM.alertSeverityBadge.className = "badge badge-ACKNOWLEDGED";
      }
      if (DOM.alertHeading) DOM.alertHeading.textContent = "INCIDENT ACKNOWLEDGED";
      if (DOM.alertStatusBadge) {
        DOM.alertStatusBadge.textContent = "ACKNOWLEDGED";
        DOM.alertStatusBadge.className = "badge badge-ACKNOWLEDGED";
      }

      // Hide Acknowledge, Reveal Resolve (Section 6)
      if (DOM.btnAcknowledge) DOM.btnAcknowledge.classList.add("hidden");
      if (DOM.btnResolve) DOM.btnResolve.classList.remove("hidden");
      if (DOM.btnPanelAck) DOM.btnPanelAck.classList.add("hidden");
      if (DOM.btnPanelResolve) DOM.btnPanelResolve.classList.remove("hidden");

      stopCountdown();
      const ackMsg = `Acknowledged (${formatTimestamp(event.acknowledged_at)})`;
      if (DOM.alertCountdownText) DOM.alertCountdownText.textContent = ackMsg;
      if (DOM.viewCountdownStatus) DOM.viewCountdownStatus.textContent = ackMsg;
      if (DOM.countdownProgressBar) DOM.countdownProgressBar.style.width = "100%";
    }

    // Populate Status & Telemetry Card
    if (DOM.noActiveEventPlaceholder) DOM.noActiveEventPlaceholder.classList.add("hidden");
    if (DOM.activeEventDetailsView) DOM.activeEventDetailsView.classList.remove("hidden");
    if (DOM.telemetryStatusBadge) DOM.telemetryStatusBadge.innerHTML = renderStatusBadge(event.status);
    if (DOM.viewEventId) DOM.viewEventId.textContent = event.event_id;
    if (DOM.viewStatusBadge) DOM.viewStatusBadge.innerHTML = renderStatusBadge(event.status);
    if (DOM.viewDetectedTime) DOM.viewDetectedTime.textContent = formatTimestamp(event.detected_at);
    if (DOM.viewOccurredTime) DOM.viewOccurredTime.textContent = event.occurred_at || "--";
    if (DOM.viewConfidence) DOM.viewConfidence.textContent = formatConfidence(event.confidence);
    if (DOM.viewSource) DOM.viewSource.textContent = event.source || "ESP32_MPU6050";

    // Location Telemetry & Map (Section 7)
    showIncidentLocationOnMap(event);

    // Re-enable interactive action buttons
    setActionButtonsDisabled(false);
  }

  function clearActiveIncident() {
    stopCountdown();
    STATE.activeIncident = null;

    // Display Normal Monitoring Panel (Section 4)
    if (DOM.noActiveIncidentSection) DOM.noActiveIncidentSection.classList.remove("hidden");
    if (DOM.activeAlertSection) DOM.activeAlertSection.classList.add("hidden");

    // Reset Active Status Card
    if (DOM.noActiveEventPlaceholder) DOM.noActiveEventPlaceholder.classList.remove("hidden");
    if (DOM.activeEventDetailsView) DOM.activeEventDetailsView.classList.add("hidden");
    if (DOM.telemetryStatusBadge) DOM.telemetryStatusBadge.innerHTML = `<span class="badge badge-monitoring">IDLE</span>`;
  }

  // --------------------------------------------------------------------------
  // Countdown Timer Engine (Response Window)
  // --------------------------------------------------------------------------
  function startCountdown(event) {
    stopCountdown();

    let elapsedSec = 0;
    try {
      const detectMs = new Date(event.detected_at).getTime();
      if (!isNaN(detectMs)) {
        elapsedSec = Math.max(0, Math.floor((Date.now() - detectMs) / 1000));
      }
    } catch {
      elapsedSec = 0;
    }

    STATE.countdownRemainingSec = Math.max(0, CONFIG.DEFAULT_RESPONSE_WINDOW_SEC - elapsedSec);
    updateCountdownDisplay();

    if (STATE.countdownRemainingSec <= 0) {
      handleCountdownExpired();
      return;
    }

    STATE.countdownIntervalId = setInterval(() => {
      STATE.countdownRemainingSec -= 1;
      updateCountdownDisplay();

      if (STATE.countdownRemainingSec <= 0) {
        stopCountdown();
        handleCountdownExpired();
      }
    }, 1000);
  }

  function updateCountdownDisplay() {
    const sec = STATE.countdownRemainingSec;
    const pct = Math.max(0, Math.min(100, (sec / CONFIG.DEFAULT_RESPONSE_WINDOW_SEC) * 100));

    if (DOM.countdownProgressBar) {
      DOM.countdownProgressBar.style.width = pct + "%";
      DOM.countdownProgressBar.style.background = pct < 25
        ? "#ef4444"
        : "linear-gradient(90deg, #ef4444, #f97316)";
    }

    if (sec > 0) {
      const text = `${sec}s remaining`;
      if (DOM.alertCountdownText) DOM.alertCountdownText.textContent = text;
      if (DOM.viewCountdownStatus) DOM.viewCountdownStatus.textContent = text;
    } else {
      if (DOM.alertCountdownText) DOM.alertCountdownText.textContent = "Response window expired";
      if (DOM.viewCountdownStatus) DOM.viewCountdownStatus.textContent = "Response window expired";
    }
  }

  function handleCountdownExpired() {
    if (DOM.alertCountdownText) DOM.alertCountdownText.textContent = "Response window expired";
    if (DOM.viewCountdownStatus) DOM.viewCountdownStatus.textContent = "Response window expired";
    showActionFeedback("Response window elapsed. Event remains confirmed pending resolution.", "error");
  }

  function stopCountdown() {
    if (STATE.countdownIntervalId) {
      clearInterval(STATE.countdownIntervalId);
      STATE.countdownIntervalId = null;
    }
  }

  // --------------------------------------------------------------------------
  // Section 7: Leaflet Map & GPS Geolocation
  // --------------------------------------------------------------------------
  function initMap() {
    if (typeof L === "undefined") {
      // Leaflet library not loaded or offline environment: render accessible fallback
      if (DOM.mapFallbackContainer) {
        DOM.mapFallbackContainer.innerHTML = `
          <div style="padding: 1.5rem; text-align: center;">
            <p style="font-weight: 700; color: #38bdf8;">GPS Geolocation Telemetry Active</p>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">Coordinates parsed directly from NEO-6M satellite stream.</p>
          </div>
        `;
      }
      return;
    }

    try {
      if (!STATE.map && DOM.incidentMap) {
        STATE.map = L.map(DOM.incidentMap, {
          zoomControl: true,
          attributionControl: false
        }).setView(CONFIG.DEFAULT_COORDINATES, 15);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 19,
          crossOrigin: true
        }).addTo(STATE.map);

        // Hide fallback container
        if (DOM.mapFallbackContainer) {
          DOM.mapFallbackContainer.style.display = "none";
        }
      }
    } catch (ex) {
      console.warn("Leaflet map initialization skipped or deferred:", ex);
    }
  }

  function showIncidentLocationOnMap(event) {
    const loc = event && event.diagnostic_metadata && event.diagnostic_metadata.location;

    // Check if valid coordinates exist
    const hasCoords = loc &&
      loc.latitude !== null &&
      loc.longitude !== null &&
      !isNaN(Number(loc.latitude)) &&
      !isNaN(Number(loc.longitude));

    // Check if coordinates are from a cached / last-known fix
    const isCached = loc && (
      loc.is_cached === true ||
      loc.cached === true ||
      loc.source === "cache" ||
      loc.status === "CACHED" ||
      loc.status === "LAST_KNOWN"
    );

    if (hasCoords) {
      const lat = Number(loc.latitude);
      const lng = Number(loc.longitude);
      const latStr = lat.toFixed(6);
      const lngStr = lng.toFixed(6);
      const sats = (loc.satellites !== undefined && loc.satellites !== null) ? loc.satellites : 0;
      const hdop = (loc.hdop !== undefined && loc.hdop !== null) ? Number(loc.hdop).toFixed(2) : "N/A";
      const mapUrl = `https://www.google.com/maps?q=${encodeURIComponent(latStr + ',' + lngStr)}`;

      // Update Alert Panel Location View
      if (DOM.alertLatitude) DOM.alertLatitude.textContent = latStr;
      if (DOM.alertLongitude) DOM.alertLongitude.textContent = lngStr;
      if (DOM.alertSatellites) DOM.alertSatellites.textContent = String(sats);
      if (DOM.alertHdop) DOM.alertHdop.textContent = String(hdop);
      if (DOM.alertMapLink) {
        DOM.alertMapLink.href = mapUrl;
        DOM.alertMapLink.classList.remove("hidden");
      }

      // Update Status Telemetry Card
      if (DOM.viewLatitude) DOM.viewLatitude.textContent = latStr;
      if (DOM.viewLongitude) DOM.viewLongitude.textContent = lngStr;
      if (DOM.viewSatellitesHdop) DOM.viewSatellitesHdop.textContent = `${sats} / ${hdop}`;
      if (DOM.viewMapRow) DOM.viewMapRow.classList.remove("hidden");
      if (DOM.viewMapLink) DOM.viewMapLink.href = mapUrl;

      // Update Map Header & Badges
      if (isCached) {
        // Cached / Last Known Location
        if (DOM.alertGpsStatusBadge) {
          DOM.alertGpsStatusBadge.textContent = "LAST KNOWN LOCATION";
          DOM.alertGpsStatusBadge.className = "badge badge-gps-invalid";
        }
        if (DOM.viewGpsStatusBadge) {
          DOM.viewGpsStatusBadge.textContent = "LAST KNOWN LOCATION";
          DOM.viewGpsStatusBadge.className = "badge badge-gps-invalid";
        }
        if (DOM.mapStatusBadge) {
          DOM.mapStatusBadge.textContent = "LAST KNOWN LOCATION";
          DOM.mapStatusBadge.className = "badge badge-gps-invalid";
        }
        if (DOM.mapLastKnownBadge) DOM.mapLastKnownBadge.classList.remove("hidden");
      } else {
        // Live GPS Fix
        if (DOM.alertGpsStatusBadge) {
          DOM.alertGpsStatusBadge.textContent = "GPS: FIX VALID";
          DOM.alertGpsStatusBadge.className = "badge badge-gps-valid";
        }
        if (DOM.viewGpsStatusBadge) {
          DOM.viewGpsStatusBadge.textContent = "FIX VALID";
          DOM.viewGpsStatusBadge.className = "badge badge-gps-valid";
        }
        if (DOM.mapStatusBadge) {
          DOM.mapStatusBadge.textContent = "GPS: FIX VALID";
          DOM.mapStatusBadge.className = "badge badge-gps-valid";
        }
        if (DOM.mapLastKnownBadge) DOM.mapLastKnownBadge.classList.add("hidden");
      }

      // Hide "GPS Unavailable" Overlay
      if (DOM.mapUnavailableOverlay) DOM.mapUnavailableOverlay.classList.add("hidden");

      // Update Map Summary & External Link
      if (DOM.mapCoordSummary) DOM.mapCoordSummary.textContent = `${latStr}, ${lngStr} (${sats} sats)`;
      if (DOM.mapExternalLink) {
        DOM.mapExternalLink.href = mapUrl;
        DOM.mapExternalLink.classList.remove("hidden");
      }

      // Update Leaflet Map Marker & Center
      if (STATE.map && typeof L !== "undefined") {
        if (STATE.currentMarker) {
          STATE.map.removeLayer(STATE.currentMarker);
          STATE.currentMarker = null;
        }

        const markerText = `
          <strong>${escapeHtml(event.event_id)}</strong><br>
          Status: ${escapeHtml(event.status)}<br>
          Confidence: ${formatConfidence(event.confidence)}<br>
          ${isCached ? '<em style="color:#f59e0b;">(Last Known Location)</em>' : '<em style="color:#22c55e;">(Live Fix)</em>'}
        `;

        STATE.currentMarker = L.marker([lat, lng])
          .addTo(STATE.map)
          .bindPopup(markerText)
          .openPopup();

        STATE.map.setView([lat, lng], 16);
      }

    } else {
      // GPS Location Unavailable (Section 7)
      if (DOM.alertLatitude) DOM.alertLatitude.textContent = "Not available";
      if (DOM.alertLongitude) DOM.alertLongitude.textContent = "Not available";
      if (DOM.alertSatellites) DOM.alertSatellites.textContent = String((loc && loc.satellites) || 0);
      if (DOM.alertHdop) DOM.alertHdop.textContent = "Not available";
      if (DOM.alertMapLink) DOM.alertMapLink.classList.add("hidden");

      if (DOM.viewLatitude) DOM.viewLatitude.textContent = "Not available";
      if (DOM.viewLongitude) DOM.viewLongitude.textContent = "Not available";
      if (DOM.viewSatellitesHdop) DOM.viewSatellitesHdop.textContent = `${(loc && loc.satellites) || 0} / Not available`;
      if (DOM.viewMapRow) DOM.viewMapRow.classList.add("hidden");

      // Badges
      if (DOM.alertGpsStatusBadge) {
        DOM.alertGpsStatusBadge.textContent = "GPS LOCATION UNAVAILABLE";
        DOM.alertGpsStatusBadge.className = "badge badge-gps-unavailable";
      }
      if (DOM.viewGpsStatusBadge) {
        DOM.viewGpsStatusBadge.textContent = "LOCATION UNAVAILABLE";
        DOM.viewGpsStatusBadge.className = "badge badge-gps-unavailable";
      }
      if (DOM.mapStatusBadge) {
        DOM.mapStatusBadge.textContent = "GPS: UNAVAILABLE";
        DOM.mapStatusBadge.className = "badge badge-gps-unavailable";
      }

      // Show Overlay, Hide Last Known Badge
      if (DOM.mapUnavailableOverlay) DOM.mapUnavailableOverlay.classList.remove("hidden");
      if (DOM.mapLastKnownBadge) DOM.mapLastKnownBadge.classList.add("hidden");

      if (DOM.mapCoordSummary) DOM.mapCoordSummary.textContent = "Location not available";
      if (DOM.mapExternalLink) DOM.mapExternalLink.classList.add("hidden");

      if (STATE.map && STATE.currentMarker) {
        STATE.map.removeLayer(STATE.currentMarker);
        STATE.currentMarker = null;
      }
    }
  }

  // --------------------------------------------------------------------------
  // Section 5 & 6: Incident Acknowledgment and Resolution Controls
  // --------------------------------------------------------------------------
  async function handleAcknowledge() {
    if (!STATE.activeIncident || STATE.inFlightAction) return;
    STATE.inFlightAction = true;
    setActionButtonsDisabled(true);

    const eventId = STATE.activeIncident.event_id;
    let success = false;

    try {
      const res = await apiFetch(`/api/fall-events/${encodeURIComponent(eventId)}/acknowledge`, {
        method: "PATCH",
        body: JSON.stringify({ acknowledged_by: "User_Dashboard" })
      });

      if (res.ok) {
        success = true;
        const data = await res.json();
        const updated = data.event;
        showActionFeedback(`Incident ${eventId} acknowledged successfully.`, "success");

        // Immediately reflect backend transition to ACKNOWLEDGED
        stopCountdown();
        STATE.activeIncident = updated;
        setActiveIncident(updated);
      } else {
        const errData = await res.json().catch(() => ({}));
        showActionFeedback(errData.message || "Failed to acknowledge incident.", "error");
        setActionButtonsDisabled(false);
      }
    } catch (err) {
      showActionFeedback("Network error: Could not reach backend to acknowledge incident.", "error");
      setActionButtonsDisabled(false);
    } finally {
      STATE.inFlightAction = false;
    }

    if (success) {
      // Background poll for history log synchronization without blocking UI
      pollEvents().catch(() => {});
    }
  }

  async function handleResolve() {
    if (!STATE.activeIncident || STATE.inFlightAction) return;
    STATE.inFlightAction = true;
    setActionButtonsDisabled(true);

    const eventId = STATE.activeIncident.event_id;
    let success = false;

    try {
      const res = await apiFetch(`/api/fall-events/${encodeURIComponent(eventId)}/resolve`, {
        method: "PATCH",
        body: JSON.stringify({ resolved_by: "User_Dashboard" })
      });

      if (res.ok) {
        success = true;
        if (STATE.resolvedEventIds) STATE.resolvedEventIds.add(eventId);
        showActionFeedback(`Incident ${eventId} resolved successfully.`, "success");

        // Incident resolved: immediately clear active emergency panel
        clearActiveIncident();
      } else {
        const errData = await res.json().catch(() => ({}));
        showActionFeedback(errData.message || "Failed to resolve incident.", "error");
        setActionButtonsDisabled(false);
      }
    } catch (err) {
      showActionFeedback("Network error: Could not reach backend to resolve incident.", "error");
      setActionButtonsDisabled(false);
    } finally {
      STATE.inFlightAction = false;
    }

    if (success) {
      // Background poll for history log synchronization without blocking UI
      pollEvents().catch(() => {});
    }
  }

  async function handleConfirmCancel() {
    if (!STATE.activeIncident || STATE.inFlightAction) return;
    STATE.inFlightAction = true;
    setActionButtonsDisabled(true);

    const eventId = STATE.activeIncident.event_id;
    const reasonText = (DOM.cancelReasonInput.value || "").trim() || "User confirmed safe / false alarm";
    let success = false;

    try {
      const res = await apiFetch(`/api/fall-events/${encodeURIComponent(eventId)}/cancel`, {
        method: "PATCH",
        body: JSON.stringify({
          cancelled_by: "User_Dashboard",
          reason: reasonText
        })
      });

      if (res.ok) {
        success = true;
        if (STATE.cancelledEventIds) STATE.cancelledEventIds.add(eventId);
        showActionFeedback(`Alert cancelled (${reasonText}).`, "success");
        if (DOM.cancelReasonPanel) DOM.cancelReasonPanel.classList.add("hidden");

        // Immediately clear active emergency panel
        clearActiveIncident();
      } else {
        const errData = await res.json().catch(() => ({}));
        showActionFeedback(errData.message || "Failed to cancel incident.", "error");
        setActionButtonsDisabled(false);
      }
    } catch (err) {
      showActionFeedback("Network error: Could not reach backend to cancel alert.", "error");
      setActionButtonsDisabled(false);
    } finally {
      STATE.inFlightAction = false;
    }

    if (success) {
      // Background poll for history log synchronization without blocking UI
      pollEvents().catch(() => {});
    }
  }

  function setActionButtonsDisabled(disabled) {
    if (DOM.btnAcknowledge) DOM.btnAcknowledge.disabled = disabled;
    if (DOM.btnResolve) DOM.btnResolve.disabled = disabled;
    if (DOM.btnOpenCancelDialog) DOM.btnOpenCancelDialog.disabled = disabled;
    if (DOM.btnConfirmCancel) DOM.btnConfirmCancel.disabled = disabled;
    if (DOM.btnPanelAck) DOM.btnPanelAck.disabled = disabled;
    if (DOM.btnPanelResolve) DOM.btnPanelResolve.disabled = disabled;
    if (DOM.btnPanelCancel) DOM.btnPanelCancel.disabled = disabled;
  }

  function showActionFeedback(message, type) {
    if (!DOM.alertActionFeedback) return;
    DOM.alertActionFeedback.textContent = message;
    DOM.alertActionFeedback.className = `action-feedback ${type}`;
    DOM.alertActionFeedback.classList.remove("hidden");
    setTimeout(() => {
      if (DOM.alertActionFeedback) DOM.alertActionFeedback.classList.add("hidden");
    }, 4500);
  }

  // --------------------------------------------------------------------------
  // Section 8: Incident History & Audit Log
  // --------------------------------------------------------------------------
  function renderHistory(events) {
    if (DOM.historyLoading) DOM.historyLoading.classList.add("hidden");

    if (!events || events.length === 0) {
      if (DOM.historyTable) DOM.historyTable.classList.add("hidden");
      if (DOM.historyEmpty) DOM.historyEmpty.classList.remove("hidden");
      if (DOM.historyError) DOM.historyError.classList.add("hidden");
      if (DOM.paginationInfo) DOM.paginationInfo.textContent = "Page 1 of 1";
      if (DOM.btnPrevPage) DOM.btnPrevPage.disabled = true;
      if (DOM.btnNextPage) DOM.btnNextPage.disabled = true;
      return;
    }

    if (DOM.historyTable) DOM.historyTable.classList.remove("hidden");
    if (DOM.historyEmpty) DOM.historyEmpty.classList.add("hidden");
    if (DOM.historyError) DOM.historyError.classList.add("hidden");

    if (DOM.historyTableBody) {
      DOM.historyTableBody.innerHTML = events.map(evt => {
        const loc = evt.diagnostic_metadata && evt.diagnostic_metadata.location;
        const hasFix = loc && loc.latitude !== null && loc.longitude !== null;
        const isCached = hasFix && (loc.is_cached === true || loc.cached === true || loc.status === "CACHED" || loc.status === "LAST_KNOWN");
        
        let gpsBadge = `<span class="badge badge-gps-unavailable" title="No GPS fix">No Fix</span>`;
        if (hasFix && isCached) {
          gpsBadge = `<span class="badge badge-gps-invalid" title="Last Known Location">Cached</span>`;
        } else if (hasFix) {
          gpsBadge = `<span class="badge badge-gps-valid" title="Valid Live GPS fix">Fix Valid</span>`;
        }

        const isSelected = STATE.selectedIncident && STATE.selectedIncident.event_id === evt.event_id;

        return `
          <tr data-event-id="${escapeHtml(evt.event_id)}" class="${isSelected ? 'row-selected' : ''}">
            <td>${escapeHtml(formatTimestamp(evt.detected_at))}</td>
            <td><strong>${escapeHtml(evt.event_id)}</strong></td>
            <td>${escapeHtml(formatConfidence(evt.confidence))}</td>
            <td>${renderStatusBadge(evt.status)}</td>
            <td>${gpsBadge}</td>
            <td>${escapeHtml(evt.source || "ESP32_MPU6050")}</td>
            <td>
              <div style="display:flex; gap:0.4rem;">
                <button class="btn btn-sm btn-outline btn-locate-map" data-event-id="${escapeHtml(evt.event_id)}" title="Show on Map">
                  Map
                </button>
                <button class="btn btn-sm btn-outline btn-view-detail" data-event-id="${escapeHtml(evt.event_id)}" title="View Full Diagnostics">
                  Details
                </button>
              </div>
            </td>
          </tr>
        `;
      }).join("");

      // Row Selection & Locate Map Listeners
      DOM.historyTableBody.querySelectorAll(".btn-locate-map").forEach(btn => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          const id = btn.getAttribute("data-event-id");
          const targetEvt = events.find(x => x.event_id === id);
          if (targetEvt) {
            STATE.selectedIncident = targetEvt;
            showIncidentLocationOnMap(targetEvt);
            renderHistory(events);
          }
        });
      });

      DOM.historyTableBody.querySelectorAll(".btn-view-detail").forEach(btn => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          const id = btn.getAttribute("data-event-id");
          openEventDetailModal(id);
        });
      });

      DOM.historyTableBody.querySelectorAll("tr").forEach(row => {
        row.addEventListener("click", () => {
          const id = row.getAttribute("data-event-id");
          const targetEvt = events.find(x => x.event_id === id);
          if (targetEvt) {
            STATE.selectedIncident = targetEvt;
            showIncidentLocationOnMap(targetEvt);
            renderHistory(events);
          }
        });
      });
    }

    // Pagination Controls
    const totalPages = Math.max(1, Math.ceil(STATE.totalEventsCount / CONFIG.PAGE_SIZE));
    if (DOM.paginationInfo) DOM.paginationInfo.textContent = `Page ${STATE.currentPage} of ${totalPages}`;
    if (DOM.btnPrevPage) DOM.btnPrevPage.disabled = STATE.currentPage <= 1;
    if (DOM.btnNextPage) DOM.btnNextPage.disabled = STATE.currentPage >= totalPages;
  }

  // --------------------------------------------------------------------------
  // Section 9: Incident Diagnostic Detail View
  // --------------------------------------------------------------------------
  async function openEventDetailModal(eventId) {
    if (!DOM.eventDetailModal || !DOM.modalContent) return;

    DOM.modalContent.innerHTML = `
      <div class="loading-state">
        <div class="spinner"></div>
        <span>Loading incident diagnostics...</span>
      </div>
    `;
    DOM.eventDetailModal.classList.remove("hidden");

    try {
      const res = await apiFetch(`/api/fall-events/${encodeURIComponent(eventId)}`);
      if (!res.ok) {
        DOM.modalContent.innerHTML = `<div class="empty-state-placeholder error-state"><p>Incident not found or failed to load.</p></div>`;
        return;
      }

      const data = await res.json();
      const evt = data.event;

      let locHtml = "";
      const modalLoc = evt.diagnostic_metadata && evt.diagnostic_metadata.location;
      if (modalLoc && modalLoc.latitude !== null && modalLoc.longitude !== null) {
        const mLat = Number(modalLoc.latitude).toFixed(6);
        const mLng = Number(modalLoc.longitude).toFixed(6);
        const mSats = modalLoc.satellites !== undefined ? modalLoc.satellites : 0;
        const mHdop = modalLoc.hdop !== undefined ? Number(modalLoc.hdop).toFixed(2) : "N/A";
        const isCached = modalLoc.is_cached === true || modalLoc.cached === true || modalLoc.status === "CACHED" || modalLoc.status === "LAST_KNOWN";
        const mMapUrl = `https://www.google.com/maps?q=${encodeURIComponent(mLat + ',' + mLng)}`;

        locHtml = `
          <div class="location-modal-box">
            <div class="location-section-header">
              <span class="location-heading">GPS Coordinates</span>
              <span class="badge ${isCached ? 'badge-gps-invalid' : 'badge-gps-valid'}">
                ${isCached ? 'LAST KNOWN LOCATION' : 'FIX VALID'}
              </span>
            </div>
            <p style="margin: 0.25rem 0;"><strong>Coordinates:</strong> ${escapeHtml(mLat)}, ${escapeHtml(mLng)}</p>
            <p style="margin: 0.25rem 0;"><strong>Satellites:</strong> ${escapeHtml(String(mSats))} | <strong>HDOP:</strong> ${escapeHtml(String(mHdop))}</p>
            <p style="margin: 0.5rem 0 0 0;"><a href="${escapeHtml(mMapUrl)}" target="_blank" rel="noopener noreferrer" class="map-link-inline">Open in Google Maps</a></p>
          </div>
        `;
      } else {
        locHtml = `
          <div class="location-modal-box">
            <div class="location-section-header">
              <span class="location-heading">GPS Coordinates</span>
              <span class="badge badge-gps-unavailable">LOCATION UNAVAILABLE</span>
            </div>
            <p style="margin: 0.25rem 0; color: #94a3b8;">No valid satellite coordinates recorded for this incident.</p>
          </div>
        `;
      }

      let metaHtml = "";
      if (evt.diagnostic_metadata) {
        metaHtml = `
          <div class="prop-row">
            <dt>Diagnostic Metadata:</dt>
            <dd><pre class="metadata-block">${escapeHtml(JSON.stringify(evt.diagnostic_metadata, null, 2))}</pre></dd>
          </div>
        `;
      }

      DOM.modalContent.innerHTML = `
        <dl class="property-list">
          <div class="prop-row"><dt>Incident ID:</dt><dd><strong>${escapeHtml(evt.event_id)}</strong></dd></div>
          <div class="prop-row"><dt>Status:</dt><dd>${renderStatusBadge(evt.status)}</dd></div>
          <div class="prop-row"><dt>Detected At:</dt><dd>${escapeHtml(evt.detected_at)}</dd></div>
          <div class="prop-row"><dt>Occurred At:</dt><dd>${escapeHtml(evt.occurred_at || "--")}</dd></div>
          <div class="prop-row"><dt>Confidence:</dt><dd>${escapeHtml(formatConfidence(evt.confidence))}</dd></div>
          <div class="prop-row"><dt>Sensor Source:</dt><dd>${escapeHtml(evt.source || "ESP32_MPU6050")}</dd></div>
          <div class="prop-row"><dt>Session ID:</dt><dd>${escapeHtml(evt.session_id || "--")}</dd></div>
          ${locHtml}
          <div class="prop-row"><dt>Created In DB:</dt><dd>${escapeHtml(evt.created_at)}</dd></div>
          <div class="prop-row"><dt>Last Updated:</dt><dd>${escapeHtml(evt.updated_at)}</dd></div>
          ${evt.acknowledged_at ? `<div class="prop-row"><dt>Acknowledged:</dt><dd>${escapeHtml(evt.acknowledged_at)} by ${escapeHtml(evt.acknowledged_by || "--")}</dd></div>` : ""}
          ${evt.resolved_at ? `<div class="prop-row"><dt>Resolved:</dt><dd>${escapeHtml(evt.resolved_at)} by ${escapeHtml(evt.resolved_by || "--")}</dd></div>` : ""}
          ${evt.cancelled_at ? `<div class="prop-row"><dt>Cancelled:</dt><dd>${escapeHtml(evt.cancelled_at)} by ${escapeHtml(evt.cancelled_by || "--")} (Reason: ${escapeHtml(evt.cancellation_reason || "--")})</dd></div>` : ""}
          ${metaHtml}
        </dl>
      `;
    } catch (err) {
      DOM.modalContent.innerHTML = `<div class="empty-state-placeholder error-state"><p>Connection error loading incident details.</p></div>`;
    }
  }

  function closeModal() {
    if (DOM.eventDetailModal) DOM.eventDetailModal.classList.add("hidden");
    if (DOM.modalContent) DOM.modalContent.innerHTML = "";
  }

  // --------------------------------------------------------------------------
  // Event Listeners & Lifecycle Setup
  // --------------------------------------------------------------------------
  function setupEventListeners() {
    // Acknowledge & Resolve actions (Section 5 & 6)
    if (DOM.btnAcknowledge) DOM.btnAcknowledge.addEventListener("click", handleAcknowledge);
    if (DOM.btnPanelAck) DOM.btnPanelAck.addEventListener("click", handleAcknowledge);

    if (DOM.btnResolve) DOM.btnResolve.addEventListener("click", handleResolve);
    if (DOM.btnPanelResolve) DOM.btnPanelResolve.addEventListener("click", handleResolve);

    // Cancel Dialog
    if (DOM.btnOpenCancelDialog) {
      DOM.btnOpenCancelDialog.addEventListener("click", () => {
        if (DOM.cancelReasonPanel) DOM.cancelReasonPanel.classList.remove("hidden");
        if (DOM.cancelReasonInput) DOM.cancelReasonInput.focus();
      });
    }

    if (DOM.btnPanelCancel) {
      DOM.btnPanelCancel.addEventListener("click", () => {
        if (DOM.activeAlertSection) DOM.activeAlertSection.classList.remove("hidden");
        if (DOM.cancelReasonPanel) DOM.cancelReasonPanel.classList.remove("hidden");
        if (DOM.cancelReasonInput) DOM.cancelReasonInput.focus();
      });
    }

    if (DOM.btnDismissCancel) {
      DOM.btnDismissCancel.addEventListener("click", () => {
        if (DOM.cancelReasonPanel) DOM.cancelReasonPanel.classList.add("hidden");
      });
    }

    if (DOM.btnConfirmCancel) DOM.btnConfirmCancel.addEventListener("click", handleConfirmCancel);

    // Manual Refresh
    if (DOM.btnRefreshHistory) {
      DOM.btnRefreshHistory.addEventListener("click", () => {
        if (DOM.historyLoading) DOM.historyLoading.classList.remove("hidden");
        pollEvents();
      });
    }

    if (DOM.btnRetryConnection) {
      DOM.btnRetryConnection.addEventListener("click", () => {
        pollSystemHealth();
        pollEvents();
      });
    }

    // Pagination
    if (DOM.btnPrevPage) {
      DOM.btnPrevPage.addEventListener("click", () => {
        if (STATE.currentPage > 1) {
          STATE.currentPage -= 1;
          pollEvents();
        }
      });
    }

    if (DOM.btnNextPage) {
      DOM.btnNextPage.addEventListener("click", () => {
        STATE.currentPage += 1;
        pollEvents();
      });
    }

    // Modal Events
    if (DOM.btnCloseModal) DOM.btnCloseModal.addEventListener("click", closeModal);
    if (DOM.btnModalClose) DOM.btnModalClose.addEventListener("click", closeModal);
    if (DOM.eventDetailModal) {
      DOM.eventDetailModal.addEventListener("click", (e) => {
        if (e.target === DOM.eventDetailModal) closeModal();
      });
    }
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && DOM.eventDetailModal && !DOM.eventDetailModal.classList.contains("hidden")) {
        closeModal();
      }
    });
  }

  // --------------------------------------------------------------------------
  // Application Entry Point
  // --------------------------------------------------------------------------
  function init() {
    initMap();
    setupEventListeners();

    // Initial fetch
    pollSystemHealth();
    pollEvents();

    // Scheduled Polling Intervals (Section 12: Controlled auto-refresh)
    if (STATE.healthIntervalId) clearInterval(STATE.healthIntervalId);
    STATE.healthIntervalId = setInterval(pollSystemHealth, CONFIG.HEALTH_POLL_INTERVAL_MS);

    if (STATE.eventIntervalId) clearInterval(STATE.eventIntervalId);
    STATE.eventIntervalId = setInterval(pollEvents, CONFIG.EVENT_POLL_INTERVAL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

})();
