document.addEventListener("DOMContentLoaded", function () {
    function addButtonsWhenReady() {
      const $table = $('#subset-example');

      if ($.fn.dataTable.isDataTable($table)) {
        const table = $table.DataTable();

        // -------------------------------
        // Hide columns
        // -------------------------------
        table.column(3).visible(false); // Severity
        table.column(4).visible(false); // Type

        // -------------------------------
        // Create custom wrapper for filters
        // -------------------------------
        const filterContainer = table.table().container().querySelector('.dataTables_filter');

        let customWrapper = document.getElementById('custom-filters-wrapper');
        if (!customWrapper) {
          customWrapper = document.createElement('div');
          customWrapper.id = 'custom-filters-wrapper';
          customWrapper.style.display = 'flex';
          customWrapper.style.alignItems = 'center';
          customWrapper.style.gap = '15px';

          // Move search input into wrapper
          const searchInput = filterContainer.querySelector('label');
          if (searchInput) customWrapper.appendChild(searchInput);

          filterContainer.appendChild(customWrapper);
        }

        // -------------------------------
        // Severity buttons
        // -------------------------------
        if (!document.getElementById('severity-label')) {
          const severityWrapper = document.createElement('div');
          severityWrapper.style.display = 'flex';
          severityWrapper.style.alignItems = 'center';
          severityWrapper.style.gap = '5px';

          const severityLabel = document.createElement('span');
          severityLabel.id = 'severity-label';
          severityLabel.textContent = 'Severity:';
          severityWrapper.appendChild(severityLabel);

          let activeFilter = null;

          new $.fn.dataTable.Buttons(table, {
            buttons: [
              {
                text: 'High',
                className: 'severity-btn high custom-btn',
                action: function (e, dt, node) {
                  if (activeFilter === 'high') {
                    dt.column(3).search('').draw();
                    activeFilter = null;
                    $(node).removeClass('active');
                  } else {
                    dt.column(3).search('^high$', true, false).draw();
                    activeFilter = 'high';
                    $('.severity-btn').removeClass('active');
                    $(node).addClass('active');
                  }
                }
              },
              {
                text: 'Low',
                className: 'severity-btn low custom-btn',
                action: function (e, dt, node) {
                  if (activeFilter === 'low') {
                    dt.column(3).search('').draw();
                    activeFilter = null;
                    $(node).removeClass('active');
                  } else {
                    dt.column(3).search('^low$', true, false).draw();
                    activeFilter = 'low';
                    $('.severity-btn').removeClass('active');
                    $(node).addClass('active');
                  }
                }
              }
            ]
          });

          table.buttons().container().appendTo(severityWrapper);
          customWrapper.appendChild(severityWrapper);
        }

        // -------------------------------
        // Dropdown for Type column (5th column)
        // -------------------------------
        if (!document.getElementById('type-label')) {
          const dropdownWrapper = document.createElement('div');
          dropdownWrapper.style.display = 'flex';
          dropdownWrapper.style.alignItems = 'center';
          dropdownWrapper.style.gap = '5px';

          const typeLabel = document.createElement('span');
          typeLabel.id = 'type-label';
          typeLabel.textContent = 'Type:';
          dropdownWrapper.appendChild(typeLabel);

          const dropdown = document.createElement('select');
          dropdown.id = 'column6-filter';
          dropdown.style.marginLeft = '5px';
          dropdown.innerHTML = '<option value="">All</option>';

          table.column(4).data().map(d => {
            const div = document.createElement('div');
            div.innerHTML = d;
            return div.textContent.trim();
          }).unique().sort().each(d => {
            dropdown.innerHTML += `<option value="${d}">${d}</option>`;
          });

          dropdown.addEventListener('change', function () {
            const val = this.value;
            if (val) table.column(4).search('^' + val + '$', true, false).draw();
            else table.column(4).search('').draw();
          });

          dropdownWrapper.appendChild(dropdown);
          customWrapper.appendChild(dropdownWrapper);
        }

        // -------------------------------
        // Reset Link (only visible when needed)
        // -------------------------------
        if (!document.getElementById('reset-filters-link')) {
          const resetWrapper = document.createElement('div');
          resetWrapper.style.display = 'flex';
          resetWrapper.style.alignItems = 'center';
          resetWrapper.style.gap = '5px';

          const resetLink = document.createElement('span'); // ✅ use <span> to avoid DataTables button styling
          resetLink.id = 'reset-filters-link';
          resetLink.textContent = 'Reset';
          resetLink.style.display = 'none'; // hidden by default
          resetLink.style.color = '#007bff';
          resetLink.style.textDecoration = 'underline';
          resetLink.style.cursor = 'pointer';
          resetLink.style.fontSize = '0.9em';
          resetLink.style.userSelect = 'none';

          // --- click handler ---
          resetLink.addEventListener('click', function () {
            // 1. Clear global search
            const searchBox = customWrapper.querySelector('input[type="search"]');
            if (searchBox) {
              searchBox.value = '';
              table.search('').draw();
            }

            // 2. Reset severity
            table.column(3).search('').draw();
            $('.severity-btn').removeClass('active');

            // 3. Reset dropdown
            const dropdown = document.getElementById('column6-filter');
            if (dropdown) {
              dropdown.value = '';
              table.column(4).search('').draw();
            }

            // 4. Hide link again
            resetLink.style.display = 'none';
          });

          resetWrapper.appendChild(resetLink);
          customWrapper.appendChild(resetWrapper);

          // -------------------------------
          // Function to show/hide the link
          // -------------------------------
          function updateResetVisibility() {
            const searchActive = table.search().length > 0;
            const severityActive = $('.severity-btn.active').length > 0;
            const dropdown = document.getElementById('column6-filter');
            const typeActive = dropdown && dropdown.value !== '';

            if (searchActive || severityActive || typeActive) {
              resetLink.style.display = 'inline';
            } else {
              resetLink.style.display = 'none';
            }
          }

          // Run the check whenever the table redraws (after filtering)
          table.on('draw', updateResetVisibility);

          // Also trigger on manual filter changes
          $('#column6-filter').on('change', updateResetVisibility);
          $(document).on('click', '.severity-btn', updateResetVisibility);
          $(document).on('input', '#custom-filters-wrapper input[type="search"]', updateResetVisibility);
        }






        return true;
      }

      return false;
    }

    const interval = setInterval(() => {
      if (addButtonsWhenReady()) clearInterval(interval);
    }, 200);
  });

// ============================================
// Resizable Sidebar Functionality
// ============================================
document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.querySelector('.bd-sidebar-primary');

  if (!sidebar) return;

  // Create resize handle and append to sidebar
  const resizeHandle = document.createElement('div');
  resizeHandle.className = 'sidebar-resize-handle';
  sidebar.appendChild(resizeHandle);

  let isResizing = false;
  let startX = 0;
  let startWidth = 0;

  // Load saved width from localStorage (only on desktop >= 1200px)
  function applySavedWidth() {
    if (window.innerWidth >= 1200) {
      const savedWidth = localStorage.getItem('sidebarWidth');
      if (savedWidth) {
        sidebar.style.width = savedWidth + 'px';
      }
    } else {
      // Remove inline width on smaller screens to let CSS media queries and theme handle it
      sidebar.style.width = '';
    }
  }

  applySavedWidth();

  // Update on window resize
  window.addEventListener('resize', applySavedWidth);

  resizeHandle.addEventListener('mousedown', function(e) {
    isResizing = true;
    startX = e.clientX;
    startWidth = parseInt(getComputedStyle(sidebar).width, 10);
    resizeHandle.classList.add('resizing');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', function(e) {
    if (!isResizing) return;

    const width = startWidth + (e.clientX - startX);
    const minWidth = 150;
    const maxWidth = 500;

    if (width >= minWidth && width <= maxWidth) {
      sidebar.style.width = width + 'px';
    }
  });

  document.addEventListener('mouseup', function() {
    if (isResizing) {
      isResizing = false;
      resizeHandle.classList.remove('resizing');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';

      // Save width to localStorage
      const currentWidth = parseInt(getComputedStyle(sidebar).width, 10);
      localStorage.setItem('sidebarWidth', currentWidth);
    }
  });
});
