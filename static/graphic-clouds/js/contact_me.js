$(function() {

  $("#contactForm input,#contactForm textarea").jqBootstrapValidation({
    preventSubmit: true,
    submitError: function($form, event, errors) {
      // additional error messages or events
    },
    submitSuccess: function($form, event) {
      unique_id();
      write_message('Information submitted! Starting...','success');
      my_data = sendStyle();

      event.preventDefault(); // prevent default submit behaviour
      // get values from FORM
      var name = $("input#name").val();
      var email = $("input#email").val();
      var phone = $("input#phone").val();
      var message = $("textarea#message").val();
      var firstName = name; // For Success/Failure Message
      // Check for white space in name for Success/Fail message
      if (firstName.indexOf(' ') >= 0) {
        firstName = name.split(' ').slice(0, -1).join(' ');
      }
      $this = $("#sendMessageButton");
      $this.prop("disabled", true); // Disable submit button until AJAX call is complete to prevent duplicate messages
      $.ajax({
        url: "/clouds-submit",
        type: "POST",
        data: my_data,
        cache: false,
        success: function() {
          // Success message
          write_message('Request completed successfully!','success');
          //clear all fields
          //$('#contactForm').trigger("reset");
        },
        error: function() {
          // Fail message
          write_message('Sorry, something is wrong with the server. Please try again later!','danger');
          //clear all fields
          //$('#contactForm').trigger("reset");
        },
        complete: function(jqXHR,status) {
          if (status == 'success')
          {
            write_output(JSON.parse(jqXHR.responseText));
            setTimeout(function() {
            $this.prop("disabled", false); // Re-enable submit button when AJAX call is complete
          }, 1000);
          }

        }
      });
    },
    filter: function() {
      return $(this).is(":visible");
    },
  });

  $("a[data-toggle=\"tab\"]").click(function(e) {
    e.preventDefault();
    $(this).tab("show");
  });
});

/*When clicking on Full hide fail/success boxes */
$('#name').focus(function() {
  $('#progress').html('');
});
