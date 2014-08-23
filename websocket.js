"use strict";

var wsUri = "ws://localhost:8000/ws";
var websocket = new WebSocket(wsUri);

function onOpen(evt) { 
	writeToScreen("CONNECTED");
	websocket.send("newgame");
}

function onClose(evt) { 
	writeToScreen("DISCONNECTED");
}

function onMessage(evt) {
	writeToScreen("data received:\n" + JSON.stringify(evt.data));
	setColors(evt.data);
}

function setColors(vector) {
	// colorvector = translateintocolors(vector);
	var colors = vector.split(" ");
	console.log(colors);
	// console.log
	d3.selectAll("circle").data(colors)
	.style("fill", function(d) {return d})
	.style("stroke", "black");    // set the line colour

}

function mouseoverBucket(category, bucket_id) {
	// category: ABV, IBU, sweetness, etc. 0,1,2
	// bucket_id: 0,1,2,3,4 (how many stars should we fill in)
	// fill in stars up to this bucket
	for (i = 0; i <= bucket_id; i++) { 
		document.getElementById("bucket_"+category+"_"+i).src="images/star_128.png";
	}
	// empty stars above this bucket
	for (i = bucket_id + 1; i <= 5; i++) { 
		document.getElementById("bucket_"+category+"_"+i).src="images/star_128_empty.png";
	}
}

function mouseoutBucket(category) {
	// category: ABV, IBU, sweetness, etc. 0,1,2
	// empty stars in this bucket (or reset to previous rating- store this?)
	for (i = 0; i < 5; i++) { 
		document.getElementById("bucket_"+category+"_"+i).src="images/star_128_empty.png";
	}
}

function setBucket(category, bucket_id) {
	// category: ABV, IBU, sweetness, etc
	// bucket_id: 0,1,2,3,4 (how many stars should we fill in)
	websocket.send("setBucket " + category + " " + bucket_id);
}

function onError(evt) { 
	writeToScreen('<span style="color: red;">ERROR:</span>\n' + evt.data);
}

function init() {
	websocket.onopen = function(evt) { onOpen(evt) };
	websocket.onclose = function(evt) { onClose(evt) };
	websocket.onmessage = function(evt) { onMessage(evt) };
	websocket.onerror = function(evt) { onError(evt) };
	document.getElementById("connect").onclick = function () {
		websocket.send(document.getElementById("inputTxt").value);
		console.log("click registered.");
	};


}

function writeToScreen(message) { 
	document.getElementById("msg").innerHTML = message;
}

window.onReady = function onReady(fn){
    document.body ? fn() : setTimeout(function(){ onReady(fn);},50);
};

window.onReady(init);