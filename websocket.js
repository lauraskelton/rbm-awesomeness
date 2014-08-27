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
		setColors(messageDict["data"]);
	} else if (messageDict["type"] == "circles") {
		//createD3Objects(messageDict["data"]);
		demoNetworkData();
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

function demoNetworkData() {
	data = {
		nodes: [
		{size: 10},
		{size: 5},
		{size: 2},
		{size: 3},
		{size: 30},
		{size: 40}
		],
		links: [
		{source: 0,target: 1},
		{source: 0,target: 2},
		{source: 1,target: 0},
		{source: 3,target: 0},
		{source: 4,target: 1}
		]
	}

	var mouseOverFunction = function(d) {
		var circle = d3.select(this);

		node
		.transition(500)
		.style("opacity", function(o) {
			return isConnected(o, d) ? 1.0 : 0.2 ;
		})
		.style("fill", function(o) {
			if (isConnectedAsTarget(o, d) && isConnectedAsSource(o, d) ) {
				fillcolor = 'green';
			} else if (isConnectedAsSource(o, d)) {
				fillcolor = 'red';
			} else if (isConnectedAsTarget(o, d)) {
				fillcolor = 'blue';
			} else if (isEqual(o, d)) {
				fillcolor = "hotpink";
			} else {
				fillcolor = '#000';
			}
			return fillcolor;
		});

		link
		.transition(500)
		.style("stroke-opacity", function(o) {
			return o.source === d || o.target === d ? 1 : 0.2;
		})
		.transition(500)
		.attr("marker-end", function(o) {
			return o.source === d || o.target === d ? "url(#arrowhead)" : "url()";
		});

		circle
		.transition(500)
		.attr("r", function(){ return 1.4 * node_radius(d)});
	}

	var mouseOutFunction = function() {
		var circle = d3.select(this);

		node
		.transition(500);

		link
		.transition(500);

		circle
		.transition(500)
		.attr("r", node_radius);
	}

	function isConnected(a, b) {
		return isConnectedAsTarget(a, b) || isConnectedAsSource(a, b) || a.index == b.index;
	}

	function isConnectedAsSource(a, b) {
		return linkedByIndex[a.index + "," + b.index];
	}

	function isConnectedAsTarget(a, b) {
		return linkedByIndex[b.index + "," + a.index];
	}

	function isEqual(a, b) {
		return a.index == b.index;
	}

	function tick() {
		link
		.attr("x1", function(d) { return d.source.x; })
		.attr("y1", function(d) { return d.source.y; })
		.attr("x2", function(d) { return d.target.x; })
		.attr("y2", function(d) { return d.target.y; });

		node
		.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
	}

	function node_radius(d) { return Math.pow(40.0 * d.size, 1/3); }

	var width = 1000;
	var height = 500;

	var nodes = data.nodes
	var links = data.links

	var force = d3.layout.force()
	.nodes(nodes)
	.links(links)
	.charge(-3000)
	.friction(0.6)
	.gravity(0.6)
	.size([width,height])
	.start();

	var linkedByIndex = {};
	links.forEach(function(d) {
		linkedByIndex[d.source.index + "," + d.target.index] = true;
	});

	var svg = d3.select("body").append("svg")
	.attr("width", width)
	.attr("height", height);

	var link = svg.selectAll("line")
	.data(links)
	.enter().append("line")

	var node = svg.selectAll(".node")
	.data(nodes)
	.enter().append("g")
	.attr("class", "node")
	.call(force.drag);

	node
	.append("circle")
	.attr("r", node_radius)
	.on("mouseover", mouseOverFunction)
	.on("mouseout", mouseOutFunction);

	svg
	.append("marker")
	.attr("id", "arrowhead")
  .attr("refX", 6 + 7) // Controls the shift of the arrow head along the path
  .attr("refY", 2)
  .attr("markerWidth", 6)
  .attr("markerHeight", 4)
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M 0,0 V 4 L6,2 Z");

  link
  .attr("marker-end", "url()");

  force
  .on("tick",tick);
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
	//document.getElementById("connect").onclick = function () {
	//	websocket.send("setBeer" + " " + document.getElementById("inputTxt").value);
	//	console.log("click registered.");
	//};

}

function writeToScreen(message) { 
	//document.getElementById("msg").innerHTML = message;
}

window.onReady = function onReady(fn){
    document.body ? fn() : setTimeout(function(){ onReady(fn);},50);

};

window.onReady(init);