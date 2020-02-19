var el = x => document.getElementById(x);

function LoadImage(input) {
	//이전 객체 지우기
	var upimg = el('upimg');
	if(upimg){
		document.getElementById("screen").removeChild(upimg);
	}
	var fileurl;
  var reader = new FileReader();
  reader.onload = function (e) {
		fileurl = e.target.result;
		console.log(fileurl);
		var w = window.innerWidth;
		var elem = document.createElement("img");
		elem.setAttribute("src", e.target.result);
		if(w>420){
			elem.setAttribute("style", "max-height: 300px; max-width: 400px; position: absolute; z-index: -1; 	top: 50%; left:50%;  transform:translate(-50%, -50%);");
		}
		else {
			elem.setAttribute("style", "max-height: 180px; max-width: 200px; position: absolute; z-index: -1; 	top: 50%; left:50%;  transform:translate(-50%, -50%);");
		}
		elem.setAttribute("alt", "Flower");
		elem.setAttribute("id", "upimg");
		document.getElementById("screen").appendChild(elem);
  }
  reader.readAsDataURL(input.files[0]);
  $("#screen-text").text("processing...");
	workhard();
	el("form1").submit();
}

function workhard() {
	$("#g1").removeClass("g1slow").addClass("g1fast").attr('fill', '#df0000');
	$("#g2").removeClass("g2slow").addClass("g2fast").attr('fill', '#e70000');
	$("#g3").removeClass("g3slow").addClass("g3fast").attr('fill', '#ff0000');
}

function worknorm() {
	$("#g1").addClass("g1slow").removeClass("g1fast").attr('fill', '#3D0000');
	$("#g2").addClass("g2slow").removeClass("g2fast").attr('fill', '#5E0000');
	$("#g3").addClass("g3slow").removeClass("g3fast").attr('fill', '#890000');
}