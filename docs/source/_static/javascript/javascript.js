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

function copyCode(button) {
  // Find the code text relative to the button
  const container = button.parentElement;
  const codeText = container.querySelector('code').innerText;

  // Use the Clipboard API
  navigator.clipboard.writeText(codeText).then(() => {
    // Visual feedback
    const originalText = button.innerText;
    button.innerText = 'Copied!';
    button.classList.add('copied');

    // Reset button after 2 seconds
    setTimeout(() => {
      button.innerText = originalText;
      button.classList.remove('copied');
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy: ', err);
  });
}

document.querySelectorAll('.copy-btn').forEach(button => {
  button.addEventListener('click', () => copyCode(button));
});
