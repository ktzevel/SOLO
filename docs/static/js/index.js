$(function() {
	$('#day-to-night .twentytwenty-container').waitForImages().done(function() {
		$('#day-to-night .twentytwenty-container').each(function() {
		$(this).twentytwenty({
				  default_offset_pct: 0.5, // How much of the before image is visible when the page loads
				  orientation: 'horizontal', // Orientation of the before and after images ('horizontal' or 'vertical')
				  before_label: 'Input', // Set a custom before label
				  after_label: 'SOLO', // Set a custom after label
				  no_overlay: false, //Do not show the overlay with before and after
				  move_slider_on_hover: false, // Move slider on mouse hover?
				  move_with_handle_only: false, // Allow a user to swipe anywhere on the image to control slider movement. 
				  click_to_move: false // Allow a user to click (or tap) anywhere on the image to move the slider to that location.
				});
		});
	});
});

$(function() {
	$('#nighttime .twentytwenty-container').waitForImages().done(function() {
		$('#nighttime .twentytwenty-container').each(function() {
		$(this).twentytwenty({
				  default_offset_pct: 0.5, // How much of the before image is visible when the page loads
				  orientation: 'horizontal', // Orientation of the before and after images ('horizontal' or 'vertical')
				  before_label: 'Real', // Set a custom before label
				  after_label: 'SOLO', // Set a custom after label
				  no_overlay: false, //Do not show the overlay with before and after
				  move_slider_on_hover: false, // Move slider on mouse hover?
				  move_with_handle_only: false, // Allow a user to swipe anywhere on the image to control slider movement. 
				  click_to_move: false // Allow a user to click (or tap) anywhere on the image to move the slider to that location.
				});
		});
	});
});

$(function() {
	$('#annotations .twentytwenty-container').waitForImages().done(function() {
		$('#annotations .twentytwenty-container').each(function() {
		$(this).twentytwenty({
				  default_offset_pct: 0.5, // How much of the before image is visible when the page loads
				  orientation: 'horizontal', // Orientation of the before and after images ('horizontal' or 'vertical')
				  before_label: 'Input', // Set a custom before label
				  after_label: 'Labels', // Set a custom after label
				  no_overlay: false, //Do not show the overlay with before and after
				  move_slider_on_hover: false, // Move slider on mouse hover?
				  move_with_handle_only: false, // Allow a user to swipe anywhere on the image to control slider movement. 
				  click_to_move: false // Allow a user to click (or tap) anywhere on the image to move the slider to that location.
				});
		});
	});
});