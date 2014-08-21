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
	d3.selectAll("circle").data(colors).style("fill", function(d) {return d});
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