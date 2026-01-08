// VANTA.js dots background effect for home page
document.addEventListener('DOMContentLoaded', function() {
    // Only run on home page
    if (!document.querySelector('.fixed-background')) return;
    
    // Load Three.js first, then VANTA
    const threeScript = document.createElement('script');
    threeScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js';
    threeScript.onload = function() {
        const vantaScript = document.createElement('script');
        vantaScript.src = 'https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.dots.min.js';
        vantaScript.onload = initVanta;
        document.head.appendChild(vantaScript);
    };
    document.head.appendChild(threeScript);
});

let vantaEffect;

function initVanta() {
    if (vantaEffect) {
        vantaEffect.destroy();
    }
    
    // Check for dark mode
    const darkMode = document.body.getAttribute('data-md-color-scheme') === 'slate';
    
    const color = darkMode ? 0xCECDC3 : 0x673ab7;
    const color2 = darkMode ? 0xCECDC3 : 0x673ab7;
    const backgroundColor = darkMode ? 0x1e1e1e : 0xffffff;
    const size = darkMode ? 0.50 : 1.00;

    vantaEffect = VANTA.DOTS({
        el: ".fixed-background",
        mouseControls: false,
        touchControls: false,
        gyroControls: false,
        minHeight: 200.00,
        minWidth: 200.00,
        scale: 1.00,
        scaleMobile: 1.00,
        color: color,
        color2: color2,
        backgroundColor: backgroundColor,
        size: size,
        spacing: 15.00,
        showLines: false
    });
}

// Watch for theme changes
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.attributeName === 'data-md-color-scheme') {
            if (typeof VANTA !== 'undefined') {
                initVanta();
            }
        }
    });
});

observer.observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });
