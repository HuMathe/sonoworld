window.onload = function() {
    initWindow();
    make_carousel('results-gen', gen_carousel_item_template, gen_items, 1, 1);
    make_carousel('results-recon', recon_carousel_item_template, recon_items, 1, 1);
};


document.addEventListener('DOMContentLoaded', function () {
    const dropdown = document.querySelector('.research-dropdown');
    if (!dropdown) return;

    document.addEventListener('click', function (event) {
        if (!dropdown.contains(event.target)) dropdown.open = false;
    });

    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && dropdown.open) {
            dropdown.open = false;
            dropdown.querySelector('summary').focus();
        }
    });
});
