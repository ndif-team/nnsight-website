// Single row navbar with inline navigation
document.addEventListener("DOMContentLoaded", function() {
  // Announcement banner
  const banner = document.createElement("div");
  banner.className = "nn-announcement";
  banner.innerHTML = 'Try our Logit Lens interface at <a href="https://workbench.ndif.us" target="_blank" rel="noopener noreferrer"><u>workbench.ndif.us</u></a> \u2192';
  document.body.insertBefore(banner, document.body.firstChild);

  // Navigation items
  const navItems = [
    { name: "Getting Started", href: "/getting-started/" },
    { name: "Documentation", href: "/documentation/" },
    { name: "Features", href: "/features/" },
    { name: "Tutorials", href: "/tutorials/" },
    { name: "Blog", href: "/blog/" },
    { name: "About", href: "/about/" }
  ];
  
  // Find the header inner container
  const headerInner = document.querySelector(".md-header__inner");
  const logo = document.querySelector(".md-header__button.md-logo");
  
  if (headerInner && logo) {
    // Create navigation container
    const headerNav = document.createElement("nav");
    headerNav.className = "header-nav";
    
    // Get current path for active state
    const currentPath = window.location.pathname;
    
    // Add navigation links
    navItems.forEach(item => {
      const link = document.createElement("a");
      link.href = item.href;
      link.textContent = item.name;
      
      // Check if this is the active page
      if (currentPath === item.href || 
          (item.href !== "/" && currentPath.startsWith(item.href))) {
        link.classList.add("active");
      }
      
      headerNav.appendChild(link);
    });
    
    // Insert nav after logo
    logo.parentNode.insertBefore(headerNav, logo.nextSibling);
  }
  
  // Add extra header icons (Discourse Forum, Discord)
  const headerSource = document.querySelector(".md-header__source");
  
  if (headerSource) {
    // Create container for extra links
    const extraLinks = document.createElement("div");
    extraLinks.className = "header-extra-links";
    
    // Discourse Forum link
    const discourseLink = document.createElement("a");
    discourseLink.href = "https://discuss.ndif.us/";
    discourseLink.target = "_blank";
    discourseLink.rel = "noopener";
    discourseLink.title = "Discourse Forum";
    discourseLink.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12.103 0C18.666 0 24 5.485 24 11.997c0 6.51-5.33 11.99-11.9 11.99L0 24V11.79C0 5.28 5.532 0 12.103 0zm.116 4.563a7.395 7.395 0 0 0-6.337 3.57 7.247 7.247 0 0 0-.148 7.22L4.4 19.61l4.794-1.074a7.424 7.424 0 0 0 8.136-1.39 7.256 7.256 0 0 0 1.737-7.997 7.375 7.375 0 0 0-6.84-4.585h-.008z"/></svg>`;
    
    // Discord link
    const discordLink = document.createElement("a");
    discordLink.href = "https://forms.gle/1Y6myaXYzSh3oHf56";
    discordLink.target = "_blank";
    discordLink.rel = "noopener";
    discordLink.title = "Join Discord";
    discordLink.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z"/></svg>`;
    
    extraLinks.appendChild(discourseLink);
    extraLinks.appendChild(discordLink);
    
    // Insert before the header source
    headerSource.parentNode.insertBefore(extraLinks, headerSource);

    // NDIF logo with animated gradient behind transparent areas
    const ndifLink = document.createElement("a");
    ndifLink.href = "/status/";
    ndifLink.title = "NDIF Status";
    ndifLink.className = "nn-ndif-badge";
    const ndifImg = document.createElement("img");
    ndifImg.alt = "NDIF";

    function updateNdifLogo() {
      var dark = document.body.getAttribute("data-md-color-scheme") === "slate";
      ndifImg.src = dark ? "/assets/NDIF_Dark.png" : "/assets/NDIF_Light.png";
    }
    updateNdifLogo();

    new MutationObserver(updateNdifLogo).observe(document.body, {
      attributes: true, attributeFilter: ["data-md-color-scheme"]
    });

    ndifLink.addEventListener("click", function(e) {
      e.preventDefault();
      window.location.href = ndifLink.href;
    });

    ndifLink.appendChild(ndifImg);
    headerSource.parentNode.insertBefore(ndifLink, headerSource.nextSibling);
  }
});
