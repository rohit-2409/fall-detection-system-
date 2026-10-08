/**
 * Stage 6K: Notification Monitoring & Reliability Dashboard Client Script.
 * Standalone Fall Detection System.
 * 
 * Polls read-only monitoring and delivery telemetry APIs at 2.5s intervals.
 * Zero mutations, zero manual retry triggers, zero secret exposure.
 */

(function () {
  'use strict';

  // State
  let currentFilter = '';
  let currentSearch = '';
  let currentPage = 1;
  const pageSize = 20;
  let totalItems = 0;
  let pollIntervalId = null;

  // DOM Elements
  const elValTotal = document.getElementById('valTotal');
  const elValSent = document.getElementById('valSent');
  const elValRetryWait = document.getElementById('valRetryWait');
  const elValQueued = document.getElementById('valQueued');
  const elValFailed = document.getElementById('valFailed');
  const elValDeadLetter = document.getElementById('valDeadLetter');
  const elValDuplicate = document.getElementById('valDuplicate');

  const elValQueueHealth = document.getElementById('valQueueHealth');
  const elValQueuePending = document.getElementById('valQueuePending');
  const elValQueueCapacity = document.getElementById('valQueueCapacity');

  const elValProviderName = document.getElementById('valProviderName');
  const elValProviderConfigured = document.getElementById('valProviderConfigured');
  const elValProviderStatus = document.getElementById('valProviderStatus');
  const elValLastUpdated = document.getElementById('valLastUpdated');

  const elTableBody = document.getElementById('deliveriesTableBody');
  const elLoading = document.getElementById('loadingDeliveries');
  const elEmpty = document.getElementById('emptyDeliveries');
  const elPaginationInfo = document.getElementById('paginationInfo');
  const elBtnPrev = document.getElementById('btnPrevPage');
  const elBtnNext = document.getElementById('btnNextPage');
  const elBtnRefresh = document.getElementById('btnRefreshDeliveries');
  const elInputSearch = document.getElementById('inputSearchEvent');

  const elModal = document.getElementById('notifDetailModal');
  const elModalContent = document.getElementById('modalNotifContent');
  const elBtnCloseModal = document.getElementById('btnCloseModal');
  const elBtnModalClose = document.getElementById('btnModalClose');

  /**
   * Helper formatting ISO date string to localized time.
   */
  function formatIsoTime(isoStr) {
    if (!isoStr) return '--';
    try {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return String(isoStr);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (_) {
      return String(isoStr);
    }
  }

  /**
   * Computes human-readable relative time (e.g. 'in 12s' or '3m ago').
   */
  function formatRelativeFuture(isoStr) {
    if (!isoStr) return '--';
    try {
      const target = new Date(isoStr).getTime();
      const now = Date.now();
      const diffSec = Math.round((target - now) / 1000);
      if (diffSec > 0) {
        if (diffSec < 60) return `in ${diffSec}s`;
        const mins = Math.floor(diffSec / 60);
        return `in ${mins}m ${diffSec % 60}s`;
      } else {
        return 'due now';
      }
    } catch (_) {
      return formatIsoTime(isoStr);
    }
  }

  /**
   * Returns CSS badge class for notification status.
   */
  function getStatusBadgeClass(status) {
    const st = (status || '').toUpperCase();
    switch (st) {
      case 'SENT': return 'badge-sent';
      case 'RETRY_WAIT': return 'badge-retry_wait';
      case 'QUEUED': return 'badge-queued';
      case 'SENDING': return 'badge-sending';
      case 'FAILED': return 'badge-failed';
      case 'DEAD_LETTER': return 'badge-dead_letter';
      case 'DUPLICATE_SUPPRESSED': return 'badge-duplicate_suppressed';
      case 'DISABLED': return 'badge-disabled';
      default: return 'badge-secondary';
    }
  }

  /**
   * Fetches monitoring overview snapshot from GET /api/notifications/monitoring.
   */
  async function fetchMonitoringSnapshot() {
    try {
      const res = await fetch('/api/notifications/monitoring');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data && data.status === 'ok') {
        const sum = data.summary || {};
        const q = data.queue || {};
        const prov = data.provider || {};

        // Summary counts
        if (elValTotal) elValTotal.textContent = sum.total || 0;
        if (elValSent) elValSent.textContent = sum.sent || 0;
        if (elValRetryWait) elValRetryWait.textContent = sum.retry_wait || 0;
        if (elValQueued) elValQueued.textContent = (sum.queued || 0) + (sum.sending || 0);
        if (elValFailed) elValFailed.textContent = sum.failed || 0;
        if (elValDeadLetter) elValDeadLetter.textContent = sum.dead_letter || 0;
        if (elValDuplicate) elValDuplicate.textContent = sum.duplicate_suppressed || 0;

        // Queue Health
        if (elValQueuePending) elValQueuePending.textContent = q.pending || 0;
        if (elValQueueCapacity) elValQueueCapacity.textContent = q.capacity || 100;
        if (elValQueueHealth) {
          elValQueueHealth.textContent = q.health || 'HEALTHY';
          elValQueueHealth.className = 'badge ' + (q.health === 'HEALTHY' ? 'badge-gps-valid' : 'badge-gps-invalid');
        }

        // Provider Status
        if (elValProviderName) elValProviderName.textContent = (prov.name || 'twilio').toUpperCase();
        if (elValProviderConfigured) {
          elValProviderConfigured.textContent = prov.configured ? 'Configured' : 'Disabled';
          elValProviderConfigured.className = 'badge ' + (prov.configured ? 'badge-gps-valid' : 'badge-gps-unavailable');
        }
        if (elValProviderStatus) elValProviderStatus.textContent = prov.status || 'Healthy';

        if (elValLastUpdated) {
          elValLastUpdated.textContent = new Date().toLocaleTimeString();
        }
      }
    } catch (err) {
      console.warn('Failed to fetch notification monitoring snapshot:', err);
    }
  }

  /**
   * Fetches delivery records from GET /api/notifications/deliveries.
   */
  async function fetchDeliveries() {
    if (!elTableBody) return;

    const offset = (currentPage - 1) * pageSize;
    let url = `/api/notifications/deliveries?limit=${pageSize}&offset=${offset}`;
    if (currentFilter) url += `&status=${encodeURIComponent(currentFilter)}`;
    if (currentSearch) url += `&event_id=${encodeURIComponent(currentSearch)}`;

    try {
      if (elLoading) elLoading.classList.remove('hidden');
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data && data.status === 'ok') {
        const items = data.items || [];
        totalItems = data.total || 0;
        renderDeliveriesTable(items);
        updatePagination();
      }
    } catch (err) {
      console.error('Failed to load deliveries:', err);
      if (elEmpty) elEmpty.classList.remove('hidden');
    } finally {
      if (elLoading) elLoading.classList.add('hidden');
    }
  }

  /**
   * Renders delivery audit table rows safely.
   */
  function renderDeliveriesTable(items) {
    elTableBody.innerHTML = '';

    if (!items || items.length === 0) {
      if (elEmpty) elEmpty.classList.remove('hidden');
      return;
    }

    if (elEmpty) elEmpty.classList.add('hidden');

    items.forEach((item) => {
      const tr = document.createElement('tr');
      const badgeClass = getStatusBadgeClass(item.status);
      const attemptsStr = `${item.attempt_count || 0} / ${item.max_attempts || 5}`;

      let nextAttemptDisplay = '--';
      if (item.status === 'RETRY_WAIT' && item.next_attempt_at) {
        nextAttemptDisplay = `${formatRelativeFuture(item.next_attempt_at)} (${formatIsoTime(item.next_attempt_at)})`;
      } else if (item.next_attempt_at) {
        nextAttemptDisplay = formatIsoTime(item.next_attempt_at);
      }

      tr.innerHTML = `
        <td style="font-weight: 700; color: #38bdf8;">${escapeHtml(item.event_id)}</td>
        <td><span class="badge ${badgeClass}">${escapeHtml(item.status)}</span></td>
        <td>${escapeHtml((item.provider || '').toUpperCase())}</td>
        <td><code>${escapeHtml(item.recipient_masked || '--')}</code></td>
        <td>${attemptsStr}</td>
        <td>${formatIsoTime(item.last_attempt_at)}</td>
        <td>${nextAttemptDisplay}</td>
        <td style="color: #f87171;">${escapeHtml(item.last_error || '--')}</td>
        <td>
          <button class="btn btn-sm btn-outline view-record-btn" data-event-id="${escapeHtml(item.event_id)}">Details</button>
        </td>
      `;

      elTableBody.appendChild(tr);
    });

    // Attach row button listeners
    elTableBody.querySelectorAll('.view-record-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const eventId = btn.getAttribute('data-event-id');
        openDetailModal(eventId);
      });
    });
  }

  /**
   * Updates pagination buttons and label.
   */
  function updatePagination() {
    const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
    if (elPaginationInfo) elPaginationInfo.textContent = `Page ${currentPage} of ${totalPages} (${totalItems} total)`;
    if (elBtnPrev) elBtnPrev.disabled = currentPage <= 1;
    if (elBtnNext) elBtnNext.disabled = currentPage >= totalPages;
  }

  /**
   * Opens detail modal with full delivery information.
   */
  async function openDetailModal(eventId) {
    if (!elModal || !elModalContent) return;

    elModalContent.innerHTML = '<div class="loading-state"><div class="spinner"></div><span>Loading event telemetry...</span></div>';
    elModal.classList.remove('hidden');

    try {
      const res = await fetch(`/api/notifications/events/${encodeURIComponent(eventId)}`);
      if (res.status === 404) {
        elModalContent.innerHTML = '<p class="empty-desc">No delivery record found for this event ID.</p>';
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data && data.notification) {
        const notif = data.notification;
        const badgeClass = getStatusBadgeClass(notif.status);
        elModalContent.innerHTML = `
          <dl class="property-list">
            <div class="prop-row"><dt>Event Identifier:</dt><dd style="font-weight: 700; color: #38bdf8;">${escapeHtml(notif.event_id)}</dd></div>
            <div class="prop-row"><dt>Notification Status:</dt><dd><span class="badge ${badgeClass}">${escapeHtml(notif.status)}</span></dd></div>
            <div class="prop-row"><dt>Provider:</dt><dd>${escapeHtml((notif.provider || '').toUpperCase())}</dd></div>
            <div class="prop-row"><dt>Masked Recipient:</dt><dd><code>${escapeHtml(notif.recipient_masked || '--')}</code></dd></div>
            <div class="prop-row"><dt>Dispatch Attempts:</dt><dd>${notif.attempt_count || 0} / ${notif.max_attempts || 5}</dd></div>
            <div class="prop-row"><dt>Last Attempt:</dt><dd>${formatIsoTime(notif.last_attempt_at)}</dd></div>
            <div class="prop-row"><dt>Next Attempt:</dt><dd>${notif.next_attempt_at ? formatIsoTime(notif.next_attempt_at) + ' (' + formatRelativeFuture(notif.next_attempt_at) + ')' : '--'}</dd></div>
            <div class="prop-row"><dt>Successful Dispatch:</dt><dd>${formatIsoTime(notif.dispatched_at)}</dd></div>
            <div class="prop-row"><dt>Last Error Code:</dt><dd style="color: #ef4444;">${escapeHtml(notif.last_error || 'None')}</dd></div>
            <div class="prop-row"><dt>Task Created:</dt><dd>${formatIsoTime(notif.created_at)}</dd></div>
            <div class="prop-row"><dt>Audit Updated:</dt><dd>${formatIsoTime(notif.updated_at)}</dd></div>
          </dl>
          <div style="margin-top: 1rem; display: flex; justify-content: flex-end;">
            <a href="/dashboard" class="btn btn-sm btn-primary">Open in Fall Monitor</a>
          </div>
        `;
      } else {
        elModalContent.innerHTML = '<p class="empty-desc">No delivery record found for this event ID.</p>';
      }
    } catch (err) {
      elModalContent.innerHTML = `<p class="empty-desc" style="color: #ef4444;">Error loading event: ${escapeHtml(err.message)}</p>`;
    }
  }

  function closeModal() {
    if (elModal) elModal.classList.add('hidden');
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Event Listeners
  document.querySelectorAll('.filter-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-status') || '';
      currentPage = 1;
      fetchDeliveries();
    });
  });

  if (elInputSearch) {
    let searchDebounce = null;
    elInputSearch.addEventListener('input', () => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        currentSearch = elInputSearch.value.trim();
        currentPage = 1;
        fetchDeliveries();
      }, 300);
    });
  }

  if (elBtnRefresh) {
    elBtnRefresh.addEventListener('click', () => {
      fetchMonitoringSnapshot();
      fetchDeliveries();
    });
  }

  if (elBtnPrev) {
    elBtnPrev.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        fetchDeliveries();
      }
    });
  }

  if (elBtnNext) {
    elBtnNext.addEventListener('click', () => {
      currentPage++;
      fetchDeliveries();
    });
  }

  if (elBtnCloseModal) elBtnCloseModal.addEventListener('click', closeModal);
  if (elBtnModalClose) elBtnModalClose.addEventListener('click', closeModal);

  if (elModal) {
    elModal.addEventListener('click', (e) => {
      if (e.target === elModal) closeModal();
    });
  }

  // Initial Load & 2.5s Polling Loop
  fetchMonitoringSnapshot();
  fetchDeliveries();

  pollIntervalId = setInterval(() => {
    fetchMonitoringSnapshot();
    fetchDeliveries();
  }, 2500);

})();
