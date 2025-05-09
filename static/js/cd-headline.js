jQuery(document).ready(function($){
	//set animation timing
	var animationDelay = 2500,
		//loading bar effect
		barAnimationDelay = 3800,
		barWaiting = barAnimationDelay - 3000, //3000 is the duration of the transition on the loading bar - set in the scss/css file
		//letters effect
		lettersDelay = 50,
		//type effect
		typeLettersDelay = 300,
		selectionDuration = 500,
		typeAnimationDelay = selectionDuration + 800,
		//clip effect 
		revealDuration = 1000,
		revealAnimationDelay = 1800;
	
	initHeadline();
	

	function initHeadline() {
		//insert <i> element for each letter of a changing word
		singleLetters($('.cd-headline.letters').find('b'));
		//initialise headline animation
		animateHeadline($('.cd-headline'));
	}

	function singleLetters($words) {
		$words.each(function(){
			var word = $(this),
				letters = word.text().split(''),
				selected = word.hasClass('is-visible');
			for (i in letters) {
				if(word.parents('.rotate-2').length > 0) letters[i] = '<em>' + letters[i] + '</em>';
				letters[i] = (selected) ? '<i class="in">' + letters[i] + '</i>': '<i>' + letters[i] + '</i>';
			}
		    var newLetters = letters.join('');
		    word.html(newLetters).css('opacity', 1);
		});
	}

	function animateHeadline($headlines) {
		var duration = animationDelay;
		$headlines.each(function(){
			var headline = $(this);
			
			if(headline.hasClass('loading-bar')) {
				duration = barAnimationDelay;
				setTimeout(function(){ headline.find('.cd-words-wrapper').addClass('is-loading') }, barWaiting);
			} else if (headline.hasClass('clip')){
				var spanWrapper = headline.find('.cd-words-wrapper'),
					newWidth = spanWrapper.width() + 10
				spanWrapper.css('width', newWidth);
			} else if (!headline.hasClass('type') ) {
				//assign to .cd-words-wrapper the width of its longest word
				var words = headline.find('.cd-words-wrapper b'),
					width = 0;
				words.each(function(){
					var wordWidth = $(this).width();
				    if (wordWidth > width) width = wordWidth;
				});
				headline.find('.cd-words-wrapper').css('width', width);
			};

			//trigger animation
			setTimeout(function(){ hideWord( headline.find('.is-visible').eq(0) ) }, duration);
		});
	}

	function hideWord($word) {
		var nextWord = takeNext($word);
		
		if($word.parents('.cd-headline').hasClass('type')) {
			var parentSpan = $word.parent('.cd-words-wrapper');
			parentSpan.addClass('selected').removeClass('waiting');	
			setTimeout(function(){ 
				parentSpan.removeClass('selected'); 
				$word.removeClass('is-visible').addClass('is-hidden').children('i').removeClass('in').addClass('out');
			}, selectionDuration);
			setTimeout(function(){ showWord(nextWord, typeLettersDelay) }, typeAnimationDelay);
		
		} else if($word.parents('.cd-headline').hasClass('letters')) {
			var bool = ($word.children('i').length >= nextWord.children('i').length) ? true : false;
			hideLetter($word.find('i').eq(0), $word, bool, lettersDelay);
			showLetter(nextWord.find('i').eq(0), nextWord, bool, lettersDelay);

		}  else if($word.parents('.cd-headline').hasClass('clip')) {
			// Make hiding also choppy
			var currentWidth = $word.parents('.cd-words-wrapper').width();
			var steps = 3; // Fewer steps for hiding
			var stepWidth = currentWidth / steps;
			
			function animateHideStep(step) {
				if (step <= steps) {
					var newWidth = currentWidth - (step * stepWidth);
					$word.parents('.cd-words-wrapper').animate({ width: Math.max(newWidth, 2) }, 80, function() {
						if (step < steps) {
							setTimeout(function() {
								animateHideStep(step + 1);
							}, 20);
						} else {
							switchWord($word, nextWord);
							showWord(nextWord);
						}
					});
				}
			}
			
			animateHideStep(1);

		} else if ($word.parents('.cd-headline').hasClass('loading-bar')){
			$word.parents('.cd-words-wrapper').removeClass('is-loading');
			switchWord($word, nextWord);
			setTimeout(function(){ hideWord(nextWord) }, barAnimationDelay);
			setTimeout(function(){ $word.parents('.cd-words-wrapper').addClass('is-loading') }, barWaiting);

		} else {
			switchWord($word, nextWord);
			setTimeout(function(){ hideWord(nextWord) }, animationDelay);
		}
	}

	function showWord($word, $duration) {
		if($word.parents('.cd-headline').hasClass('type')) {
			// Modified to ensure one letter at a time typing effect
			$word.addClass('is-visible').removeClass('is-hidden');
			
			// Get all letters in the word
			var $letters = $word.find('i');
			
			// Hide all letters initially
			$letters.removeClass('in').addClass('out');
			
			// Show letters one by one with delay
			var letterIndex = 0;
			
			function showNextLetter() {
				if (letterIndex < $letters.length) {
					$($letters[letterIndex]).removeClass('out').addClass('in');
					letterIndex++;
					setTimeout(showNextLetter, typeLettersDelay);
				} else {
					// Once all letters are shown, wait before hiding the word
					setTimeout(function() { hideWord($word) }, animationDelay);
				}
			}
			
			// Start the letter-by-letter animation
			showNextLetter();
		} else if($word.parents('.cd-headline').hasClass('clip')) {
			// Add this multi-step animation to create a choppy effect
			var wordWidth = $word.width() + 10;
			var steps = 5; // Number of steps for choppy effect
			var stepWidth = wordWidth / steps;
			var currentWidth = 0;
			
			// Use revealDuration for the total animation time
			var stepDuration = Math.floor(revealDuration / steps);
			var stepDelay = 30; // Pause between steps
			
			function animateStep(step) {
				if (step <= steps) {
					currentWidth = step * stepWidth;
					$word.parents('.cd-words-wrapper').animate({ 'width': currentWidth }, stepDuration, function() {
						setTimeout(function() {
							animateStep(step + 1);
						}, stepDelay);
					});
				} else {
					setTimeout(function() { hideWord($word) }, revealAnimationDelay);
				}
			}
			
			animateStep(1);
		}
	}

	function hideLetter($letter, $word, $bool, $duration) {
		$letter.removeClass('in').addClass('out');
		
		if(!$letter.is(':last-child')) {
		 	setTimeout(function(){ hideLetter($letter.next(), $word, $bool, $duration); }, $duration);  
		} else if($bool) { 
		 	setTimeout(function(){ hideWord(takeNext($word)) }, animationDelay);
		}

		if($letter.is(':last-child') && $('html').hasClass('no-csstransitions')) {
			var nextWord = takeNext($word);
			switchWord($word, nextWord);
		} 
	}

	function showLetter($letter, $word, $bool, $duration) {
		$letter.addClass('in').removeClass('out');
		
		if(!$letter.is(':last-child')) { 
			setTimeout(function(){ showLetter($letter.next(), $word, $bool, $duration); }, $duration); 
		} else { 
			if($word.parents('.cd-headline').hasClass('type')) { setTimeout(function(){ $word.parents('.cd-words-wrapper').addClass('waiting'); }, 200);}
			if(!$bool) { setTimeout(function(){ hideWord($word) }, animationDelay) }
		}
	}

	function takeNext($word) {
		return (!$word.is(':last-child')) ? $word.next() : $word.parent().children().eq(0);
	}

	function takePrev($word) {
		return (!$word.is(':first-child')) ? $word.prev() : $word.parent().children().last();
	}

	function switchWord($oldWord, $newWord) {
		$oldWord.removeClass('is-visible').addClass('is-hidden');
		$newWord.removeClass('is-hidden').addClass('is-visible');
	}
});