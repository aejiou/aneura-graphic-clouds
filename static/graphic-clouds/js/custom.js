function imToggle(im,src)
{
  document.getElementById(im).src = "img/about/"+src+".jpg";
}

function sendStyle()
{
    var elems = ['identifier','name','engine','t_im_range','transform','colors','im_width','im_height'];
    var checkboxes = ['font1','font2','font3','font4','inverted'];
    var dt = "var data = { ";
    for (i=0; i<elems.length; i++)
     {
         dt = dt + '"' + elems[i] + '" : "' + document.getElementById(elems[i]).value + '"' + ' , ';
     }
    for (i=0; i<checkboxes.length; i++)
     {
         dt = dt + '"' + checkboxes[i] + '" : "' + document.getElementById(checkboxes[i]).checked + '"' + ((i == checkboxes.length-1) ? '' : ' , ');
     }
    dt = dt + " }";
    d = eval(dt);
    return data;

}


function write_message(msg,t)
 {
    $('#progress').html("<div class='alert alert-"+t+"'>");
    //$('#progress > .alert-'+t).html("<button type='button' class='close' data-dismiss='alert' aria-hidden='true'>&times;")
    //  .append("</button>");
    $('#progress > .alert-'+t)
      .append("<strong>"+msg+"</strong>");
    $('#progress > .alert-'+t)
      .append('</div>');

 }

function unique_id()
 {
     r = "";
     for (i=0; i<16; i++)
      {
        r = r + String.fromCharCode(Math.round(Math.random()*22)+65);
      }
    document.getElementById('identifier').value = r;

 }

function write_output(result)
 {
    $('#output').html("<h2>"+result.concept+"</h2>");
    $('#output').append('    <img class="img-fluid" src="'+result.src+'" alt="">');
    $('#output').append('<p class="item-intro text-muted">'+result.caption+'</p>');
    $('#output').append('<button class="btn btn-primary" data-dismiss="modal" type="button"><i class="fas fa-times"></i>Close</button>');
 }