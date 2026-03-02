document.addEventListener("DOMContentLoaded", () => {
  // Existing tab functionality
  document.querySelectorAll('.tab-item').forEach(tab => {
    tab.addEventListener('click', () => {
      const container = tab.closest('.tabs-container');
      const targetTab = tab.getAttribute('data-tab');

      // Remove active class from all tabs and panels
      container.querySelectorAll('.tab-item')
        .forEach(t => t.classList.remove('active'));
      container.querySelectorAll('.tab-panel')
        .forEach(p => p.classList.remove('active'));

      // Activate selected tab
      tab.classList.add('active');
      document.getElementById(targetTab).classList.add('active');
    });
  });

  // Header search functionality
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        const query = this.value.trim();
        if (query) {
          // Navigate to search page with query
          const searchUrl = new URL('search.html', window.location.origin + window.location.pathname);
          searchUrl.searchParams.set('q', query);
          window.location.href = searchUrl.toString();
        }
      }
    });
  }
  
  // Header theme toggle functionality
  const themeButton = document.querySelector('.theme-switch-button');
  const themeIcon = document.getElementById('theme-icon');
  
  if (themeButton && themeIcon) {
    
    // Function to update icon based on current theme
    function updateThemeIcon() {
      const currentTheme = document.documentElement.dataset.theme || 'auto';
      const isDark = currentTheme === 'dark' || 
                    (currentTheme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches);
      
      themeIcon.className = isDark ? 'fa fa-moon' : 'fa fa-sun';
    }
    
    // Initial icon update
    updateThemeIcon();
    
    // Theme toggle click handler
    themeButton.addEventListener('click', function() {
      const currentTheme = document.documentElement.dataset.theme || 'auto';
      let newTheme;
      
      if (currentTheme === 'auto' || currentTheme === 'light') {
        newTheme = 'dark';
      } else {
        newTheme = 'light';
      }
      
      // Update theme
      document.documentElement.dataset.theme = newTheme;
      localStorage.setItem('theme', newTheme);
      
      // Update icon
      updateThemeIcon();
    });
    
    // Listen for system theme changes when in auto mode
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateThemeIcon);
  }
});

function toggleWarningDetails() {
  const details = document.getElementById('warningDetails');
  const btn = document.getElementById('expandBtn');
  const banner = document.getElementById('warningBanner');
  const mainContent = document.querySelector('.landing');
  
  if (details.classList.contains('expanded')) {
    details.classList.remove('expanded');
    btn.textContent = 'Learn More';
    
    // Reset padding to default when collapsed
    if (mainContent) {
      mainContent.style.transition = 'padding-top 0.3s ease';
      mainContent.style.paddingTop = '9rem';
    }
  } else {
    details.classList.add('expanded');
    btn.textContent = 'Show Less';
    
    // Wait for expansion animation, then adjust padding based on banner height
    setTimeout(() => {
      if (banner && mainContent) {
        const bannerHeight = banner.offsetHeight;
        const headerHeight = 73;
        const totalOffset = bannerHeight + headerHeight;
        mainContent.style.transition = 'padding-top 0.2s ease';
        mainContent.style.paddingTop = `${totalOffset + 24}px`;
      }
    }, 50);
  }
}

function closeWarningBanner() {
  const banner = document.getElementById('warningBanner');
  const mainContent = document.querySelector('.landing');
  
  banner.classList.add('hidden');
  
  // Adjust main content padding when banner is closed
  setTimeout(() => {
    if (mainContent) {
      mainContent.style.transition = 'padding-top 0.3s ease';
      mainContent.style.paddingTop = '6rem';
    }
  }, 300);
}

// Code copying functions
function copyCode(text, btn) {
    // Attempt modern Clipboard API
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text)
            .then(() => animateButton(btn))
            .catch(() => fallbackCopy(text, btn));
    } else {
        fallbackCopy(text, btn);
    }
}
function fallbackCopy(text, btn) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    
    // Ensure the textarea is off-screen
    textArea.style.position = "fixed";
    textArea.style.left = "-9999px";
    textArea.style.top = "0";
    document.body.appendChild(textArea);
    
    textArea.focus();
    textArea.select();

    try {
        const successful = document.execCommand('copy');
    } catch (err) {
        console.error('Fallback: Unable to copy', err);
    }

    document.body.removeChild(textArea);
}

function animateButton(btn) {
    const originalText = btn.innerHTML;
    
    // Change to Checkmark
    btn.innerHTML = "&#10003;"; // Unicode Checkmark
    btn.classList.add('success');
    
    // Reset after 2 seconds
    setTimeout(() => {
        btn.innerHTML = originalText;
        btn.classList.remove('success');
    }, 2000);
}

// Add click event listeners to all copy buttons
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.copy-btn').forEach(button => {
    button.addEventListener('click', () => copyCode(button));
  });
});
