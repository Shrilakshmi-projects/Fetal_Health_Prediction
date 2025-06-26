// JavaScript to handle some basic interactivity (if needed)
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector('form');
    form.addEventListener('submit', function (event) {
        // Example: Simple validation before submission
        const inputs = form.querySelectorAll('input');
        let valid = true;
        inputs.forEach(input => {
            if (input.value.trim() === '') {
                input.style.border = '2px solid red';
                valid = false;
            } else {
                input.style.border = '1px solid #ccc';
            }
        });
        if (!valid) {
            event.preventDefault();
        }
    });
});