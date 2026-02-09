document.addEventListener("DOMContentLoaded", () => {
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
