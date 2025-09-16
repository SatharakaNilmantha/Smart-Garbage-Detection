// Main JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const selectedFile = document.getElementById('selected-file');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const uploadBox = document.querySelector('.upload-box');

    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Validate file type
                const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg'];
                if (!allowedTypes.includes(file.type)) {
                    showError('Please select a valid image file (PNG, JPG, JPEG)');
                    return;
                }

                // Validate file size (max 10MB)
                const maxSize = 10 * 1024 * 1024; // 10MB
                if (file.size > maxSize) {
                    showError('File size must be less than 10MB');
                    return;
                }

                // Show selected file
                selectedFile.textContent = `Selected: ${file.name}`;
                selectedFile.style.display = 'block';
                
                // Enable submit button
                submitBtn.classList.add('active');
                
                // Update upload box appearance
                uploadBox.style.borderColor = '#48bb78';
                uploadBox.style.background = 'linear-gradient(145deg, #f0fff4, #e6fffa)';
                
                // Clear any previous errors
                clearError();
            }
        });
    }

    // Form submission handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const file = fileInput.files[0];
            if (!file) {
                e.preventDefault();
                showError('Please select a file first');
                return;
            }

            // Show loading animation
            if (loading) {
                loading.style.display = 'block';
                submitBtn.disabled = true;
                submitBtn.textContent = 'Processing...';
            }
        });
    }

    // Drag and drop functionality
    if (uploadBox) {
        uploadBox.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadBox.style.borderColor = '#667eea';
            uploadBox.style.background = 'linear-gradient(145deg, #edf2f7, #e2e8f0)';
        });

        uploadBox.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadBox.style.borderColor = '#cbd5e0';
            uploadBox.style.background = 'linear-gradient(145deg, #f7fafc, #edf2f7)';
        });

        uploadBox.addEventListener('drop', function(e) {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                fileInput.dispatchEvent(new Event('change'));
            }
            uploadBox.style.borderColor = '#cbd5e0';
            uploadBox.style.background = 'linear-gradient(145deg, #f7fafc, #edf2f7)';
        });
    }

    // Error handling functions
    function showError(message) {
        clearError();
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        
        const container = document.querySelector('.container');
        container.appendChild(errorDiv);
        
        // Reset file input
        if (fileInput) {
            fileInput.value = '';
        }
        if (selectedFile) {
            selectedFile.style.display = 'none';
        }
        if (submitBtn) {
            submitBtn.classList.remove('active');
        }
    }

    function clearError() {
        const existingError = document.querySelector('.error');
        if (existingError) {
            existingError.remove();
        }
    }

    // Image preview functionality
    function previewImage(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            let preview = document.getElementById('image-preview');
            if (!preview) {
                preview = document.createElement('img');
                preview.id = 'image-preview';
                preview.style.maxWidth = '200px';
                preview.style.maxHeight = '200px';
                preview.style.borderRadius = '8px';
                preview.style.marginTop = '15px';
                preview.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
                selectedFile.appendChild(preview);
            }
            preview.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // Add image preview to file selection
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                previewImage(file);
            }
        });
    }

    // Smooth scrolling for result page
    if (window.location.pathname === '/result') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Add click-to-copy functionality for prediction result
    const prediction = document.querySelector('.prediction');
    if (prediction) {
        prediction.style.cursor = 'pointer';
        prediction.title = 'Click to copy';
        prediction.addEventListener('click', function() {
            navigator.clipboard.writeText(this.textContent).then(function() {
                const originalText = prediction.textContent;
                prediction.textContent = 'Copied!';
                setTimeout(function() {
                    prediction.textContent = originalText;
                }, 1000);
            });
        });
    }
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Add some visual feedback for better UX
function addRippleEffect(element) {
    element.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');
        
        element.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    });
}

// Apply ripple effect to buttons
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('.btn, .file-input-button, .submit-btn');
    buttons.forEach(addRippleEffect);
});