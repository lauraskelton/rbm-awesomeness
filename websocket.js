"use strict";

var wsUri = "ws://localhost:8000/ws";
var websocket = new WebSocket(wsUri);

var abv_bucket = -1;
var ibu_bucket = -1;
var gravity_bucket = -1;

function onOpen(evt) { 
	writeToScreen("CONNECTED");
	//websocket.send("newgame");

}

function onClose(evt) { 
	writeToScreen("DISCONNECTED");
}

function onMessage(evt) {
	writeToScreen("data received:\n" + JSON.stringify(evt.data));
		console.log(evt.data);
	var messageDict = JSON.parse(evt.data);
	if (messageDict["type"] == "colors") {
		//setColors(messageDict["data"]);
		setColors(messageDict["data"]);
	} else if (messageDict["type"] == "circles") {
		createD3Objects(messageDict["data"]);
	} else if (messageDict["type"] == "nodes") {
		demoNetworkData(messageDict["data"]);
	}
}

function createD3Objects(circleData) {

	//var circleData = [{id:1, cx:40,cy:60}, {id:2, cx:80,cy:60}, {id:3, cx:120,cy:60}]
	console.log("data below!")
	console.log(circleData)
 
	var svgContainer = d3.select("body").append("svg")
	                                     .attr("width", 720)
	                                     .attr("height", 120);
	 
	var circles = svgContainer.selectAll("circle")
										.data(circleData)
										.enter()
										.append("circle")

	var circleAttributes = circles
							.attr("cx", function (d) { return d.cx; })
							.attr("cy", function (d) { return d.cy; })
							.attr("r", 10)
							.style("fill", "white")
							.style("stroke", "black");
}

function demoNetworkData(data) {

	function node_radius(d) { return Math.pow(40.0 * d.size, 1/3); }

	var width = 1300;
	var height = 700;

	var nodes = data.nodes
	var links = data.links

	var svg = d3.select("body").append("svg")
            .attr("width", width)
            .attr("height", height);

    svg.selectAll("line")
      .data(links)
    .enter().append("line")
      .attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

  svg.selectAll("circle")
      .data(nodes)
    .enter().append("circle")
      .attr("cx", function(d) { return width * d.x})
      .attr("cy", function(d) { return 64 + (200 * d.y)})
      .attr("r", node_radius);


	// svg
	// .append("marker")
	// .attr("id", "arrowhead")
 //  .attr("refX", 6 + 7) // Controls the shift of the arrow head along the path
 //  .attr("refY", 2)
 //  .attr("markerWidth", 6)
 //  .attr("markerHeight", 4)
 //  .attr("orient", "auto")
 //  .append("path")
 //  .attr("d", "M 0,0 V 4 L6,2 Z");

 //  link
 //  .attr("marker-end", "url()");

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

function setColors(colors) {
	d3.selectAll("circle").data(colors)
	.style("fill", function(d) {return d})
	.style("stroke", "black");    // set the line colour

}

function setNodeColors(colors) {
	d3.selectAll(".node").data(colors)

		.style("fill", function(d) {return d})
		.style("stroke", "black");  
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
}

function mouseoutBucket(category) {
	// category: ABV, IBU, sweetness, etc. 0,1,2
	// empty stars in this bucket (or reset to previous rating- store this?)
	var bucket_id = -1;
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
		websocket.send("setBeer" + " " + document.getElementById("inputTxt").value);
		console.log("click registered.");
	};

}

function writeToScreen(message) { 
	//document.getElementById("msg").innerHTML = message;
}

window.onReady = function onReady(fn){
    document.body ? fn() : setTimeout(function(){ onReady(fn);},50);

};

window.onReady(init);