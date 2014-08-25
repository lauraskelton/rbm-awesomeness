"use strict";

var wsUri = "ws://localhost:8000/ws";
var websocket = new WebSocket(wsUri);

var abv_bucket = -1;
var ibu_bucket = -1;
var gravity_bucket = -1;

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

function setCategoryToBucket(category, bucket_id) {
	switch(category) {
	    case 0:
	        abv_bucket = bucket_id;
	        break;
	    case 1:
	        ibu_bucket = bucket_id;
	        break;
	    case 2:
	        gravity_bucket = bucket_id;
	        break;
	    default:
	        break;
	}
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
	for (var i = 0; i <= bucket_id; i++) { 
		document.getElementById("bucket_"+category+"_"+i).src="images/star_128.png";
	}
	// empty stars above this bucket
	for (var i = bucket_id + 1; i < 5; i++) { 
		document.getElementById("bucket_"+category+"_"+i).src="images/star_128_empty.png";
	}
	setCategoryToBucket(category, bucket_id);
}

function mouseoutBucket(category) {
	// category: ABV, IBU, sweetness, etc. 0,1,2
	// empty stars in this bucket (or reset to previous rating- store this?)
	var bucket_id = 0;
	switch(category) {
	    case 0:
	        bucket_id = abv_bucket;
	        break;
	    case 1:
	        bucket_id = ibu_bucket;
	        break;
	    case 2:
	        bucket_id = gravity_bucket;
	        break;
	    default:
	        break;
	}
	for (var i = 0; i <= bucket_id; i++) { 
		document.getElementById("bucket_"+category+"_"+i).src="images/star_128.png";
	}
	// empty stars above this bucket
	for (var i = bucket_id + 1; i < 5; i++) { 
		document.getElementById("bucket_"+category+"_"+i).src="images/star_128_empty.png";
	}
}

function setBucket(category, bucket_id) {
	// category: ABV, IBU, sweetness, etc
	// bucket_id: 0,1,2,3,4 (how many stars should we fill in)
	setCategoryToBucket(category, bucket_id);

	var outString = "setBucket"
	if (abv_bucket != -1) {
		outString += " " + "ABV" + " " + abv_bucket;
	}
	if (ibu_bucket != -1) {
		outString += " " + "IBU" + " " + ibu_bucket;
	}
	if (gravity_bucket != -1) {
		outString += " " + "GRAVITY" + " " + gravity_bucket;
	}

	websocket.send(outString);
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