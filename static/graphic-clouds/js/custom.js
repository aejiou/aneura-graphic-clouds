function imToggle(im,src)
{
  var res;
  if (im=='il_style') { 
    if (src=='d_l') { src = document.forms[0].elements['style'].value; }
    d_l = (document.forms[0]['inverted'].checked) ? '_dark' : "_light"; 
    res = src + d_l; 
  } 
  else { res = src; }
  document.getElementById(im).src = "img/about/"+ res +".jpg";
}

function fontsToggle(add){
  var adds = ['_l','_u',''];
  ext = add; 
 for (i=0; i<fonts.length; i++)
 {
   if (add == 'random') {ext = adds[Math.round(Math.random()*2)];}
   document.getElementById('fontimg'+i).src = "img/fonts/"+fonts[i]+ext+".gif";
 } 
}

function writeFonts(){
  fonts = ['antiqua','ashbury','brochurn','cloistrk', 'cushing','distress','eklektic',
 'geometr', 'hobo', 'lucian', 'motterfem', 'myriadpro', 'nuptial', 'pantspatrol', 'polaroid',
 'raleigh'];
  var html = "";
  for (i=0; i<fonts.length; i++)
  {
   html = html + '<input class="form-radio" id="'+fonts[i]+'" name="'+fonts[i]+ '"' + ( (i==0) ? ' checked' : '')+' type="checkbox"><img src=img/fonts/'+fonts[i]+'.gif id="fontimg'+i+'" width="122">';
   html = html + ((i/2 == Math.round(i/2)) ? "" : "<br>");
  }
  $('#font-selection').html(html);
 

}

function sendStyle()
{
    var elems = ['identifier','concept','keywords','mask','style','im_width','im_height'];
    var checkboxes = ['inverted'];
    //checkboxes[checkboxes.length] = 'inverted';
    var dt = "var data = { ";
    for (i=0; i<elems.length; i++)
     {
         dt = dt + '"' + elems[i] + '" : "' + document.forms[0][elems[i]].value + '"' + ' , ';
     }
    for (i=0; i<checkboxes.length; i++)
     {
         dt = dt + '"' + checkboxes[i] + '" : "' + document.getElementById(checkboxes[i]).checked + '"' + ((i == checkboxes.length-1) ? '' : ' , ');
     }
    dt = dt + " }";
    d = eval(dt);
    //fonts.pop();
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


function get_update()
  {
    $.ajax({
      url: "/clouds-status",
      type: "GET",
      data: { 'id': my_data.identifier },
      cache: false,
      complete: function(jqXHR,status) {
        if (status == 'success')
        {
          write_message(jqXHR.responseText,'success');
        }
        u_timer = setTimeout(get_update, 5000);
      }
    });

  }