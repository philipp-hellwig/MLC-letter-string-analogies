// --- Config ---
const CONFIG = {
    style: {
        colors: {
            attention: d3.interpolateViridis,
            hidden: d3.interpolateMagma,
            highlight: "#FFA500", 
            connection: "#888",   
            textMain: "#fafafa",
            textDim: "#888",
            bgBox: "rgba(255, 255, 255, 0.02)",
        },
        ui: {
            toggleWidth: 60,
            toggleHeight: 25,
            activeColor: "#FFA500",
            inactiveColor: "#444"
        },
        scaling: { // For Attention Views:
            large: { // For Averaged View
                fontSizeMult: 0.6,
                minFontSize: 10,
                separatorMult: 0.1,
                labelOffsetMult: 1.5
            },
            small: { // For Grid/Head View
                fontSizeMult: 0.45,
                minFontSize: 3,
                separatorMult: 0.08,
                labelOffsetMult: 1.5
            }
        }
    },
    layout: {
        margin: { top: 150, right: 250, bottom: 800, left: 120 },
        gutterWidth: 150, 
        verticalGap: 150  
    },
    nodes: {
        encoderBlock: {
            heatmapSize: 300,
            hiddenWidth: 900, 
            borderRadius: 15,
            gridPadding: 10,  // Gap between heads
            titleSpace: 15,   // Vertical space for "H0", "H1" etc.
        },
        decoder: {
            width: 220,
            height: 90,
            borderRadius: 12,
            topMargin: 200
        },
        predictions: {
            topMargin: 100
        }
    }
};

// --- Global State for Interaction ---
let isPerturbationMode = false;

const zoom = d3.zoom()
    .on("zoom", (event) => {
        d3.select("svg g").attr("transform", event.transform);
    });

// --- Helper Functions ---
function sendReady() {
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:componentReady",
    apiVersion: 1
  }, "*");
}

function setFrameHeight(height) {
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:setFrameHeight",
    height: height
  }, "*");
}

function sendToStreamlit(value) {
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:setComponentValue",
    value: value
  }, "*");
}

function showTooltip(event, src, dst, val) {
    const tooltip = d3.select("#tooltip");
    tooltip.transition().duration(50).style("opacity", 1);
    tooltip.html(`${src} &rarr; ${dst} : <b>${val.toFixed(2)}</b>`)
           .style("left", (event.pageX + 15) + "px")
           .style("top", (event.pageY - 20) + "px");
}

function hideTooltip() {
    d3.select("#tooltip").transition().duration(100).style("opacity", 0);
}

function drawAblationToggle(parent, x, y) {
    const { toggleWidth, toggleHeight, activeColor, inactiveColor } = CONFIG.style.ui;

    const toggleG = parent.append("g")
        .attr("transform", `translate(${x}, ${y})`)
        .style("cursor", "pointer")
        .on("click", function() {
            isPerturbationMode = !isPerturbationMode;
            updateToggle();
        });

    toggleG.append("text")
        .attr("x", -10).attr("y", toggleHeight / 2)
        .attr("text-anchor", "end").attr("alignment-baseline", "middle")
        .style("fill", "white").style("font-size", "14px").text("Perturbation Mode (only supports perturbations at hidden layer that connects to the decoder)");

    const track = toggleG.append("rect")
        .attr("width", toggleWidth).attr("height", toggleHeight)
        .attr("rx", toggleHeight / 2)
        .attr("fill", isPerturbationMode ? activeColor : inactiveColor);

    const knob = toggleG.append("circle")
        .attr("cx", isPerturbationMode ? toggleWidth - (toggleHeight / 2) : toggleHeight / 2)
        .attr("cy", toggleHeight / 2).attr("r", (toggleHeight / 2) - 2).attr("fill", "white");

    function updateToggle() {
        // Visual Update
        track.transition().duration(200).attr("fill", isPerturbationMode ? activeColor : inactiveColor);
        knob.transition().duration(200).attr("cx", isPerturbationMode ? toggleWidth - (toggleHeight / 2) : toggleHeight / 2);
        
        const svg = d3.select("svg");
        const brushes = d3.selectAll(".brush, .brush .overlay");

        if (isPerturbationMode) {
            // --- ABLATION MODE ---
            svg.on(".zoom", null); // Kill zoom listeners
            brushes.style("pointer-events", "all");
            svg.style("cursor", "crosshair");
        } else {
            // --- PANNING MODE ---
            svg.call(zoom); // Re-attach zoom
            // Re-attach double click specifically
            svg.on("dblclick.zoom", function(event) {
                svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
            });
            brushes.style("pointer-events", "none");
            svg.style("cursor", "default");
        }
    }
    updateToggle();
}

function drawHeatmap(g, matrix, labels, cellSize, showYLabels, separators, sizeMode = "large") {
    const activeSeparators = separators || [];
    const flatValues = matrix.flat();
    const dynamicDomain = d3.extent(flatValues);
    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain(dynamicDomain);

    // Pull scaling params from CONFIG based on sizeMode ("small" or "large")
    const cfg = CONFIG.style.scaling[sizeMode];
    
    const labelFontSize = Math.max(cellSize * cfg.fontSizeMult, cfg.minFontSize);
    const separatorWidth = Math.max(cellSize * cfg.separatorMult, 0.5);
    const labelOffset = cellSize * cfg.labelOffsetMult;

    matrix.forEach((row, i) => {
        if (showYLabels) {
            g.append("text")
                .attr("x", -5)
                .attr("y", i * cellSize + cellSize / 2)
                .attr("text-anchor", "end")
                .attr("alignment-baseline", "middle")
                .style("fill", "#fafafa")
                .style("font-size", `${labelFontSize}px`)
                .text(labels[i]);
        }

        row.forEach((value, j) => {
            if (i === matrix.length - 1) {
                g.append("text")
                    .attr("x", j * cellSize + cellSize / 2)
                    .attr("y", (i + 1) * cellSize + labelOffset)
                    .attr("text-anchor", "middle")
                    .style("fill", "#fafafa")
                    .style("font-size", `${labelFontSize}px`)
                    .text(labels[j]);
            }

            g.append("rect")
                .attr("x", j * cellSize)
                .attr("y", i * cellSize)
                .attr("width", cellSize) 
                .attr("height", cellSize)
                .style("fill", colorScale(value))
                .on("mouseover", (event) => {
                    if (isPerturbationMode) return;
                    const selection = d3.select(event.currentTarget);
                    d3.select(event.currentTarget).style("stroke", "white").style("stroke-width", 1);
                    showTooltip(event, labels[j], labels[i], value);
                })
                .on("mouseout", (event) => {
                    d3.select(event.currentTarget).style("stroke", "none");
                    hideTooltip();
                });
        });
    });

    activeSeparators.forEach(pos => {
        const linePos = pos * cellSize;
        const dashArray = `${cellSize/4},${cellSize/8}`;

        g.append("line")
            .attr("x1", linePos).attr("y1", 0)
            .attr("x2", linePos).attr("y2", matrix.length * cellSize)
            .attr("stroke", "#FFA500")
            .attr("stroke-width", separatorWidth)
            .attr("stroke-dasharray", dashArray);

        g.append("line")
            .attr("x1", 0).attr("y1", linePos)
            .attr("x2", matrix[0].length * cellSize).attr("y2", linePos)
            .attr("stroke", "#FFA500")
            .attr("stroke-width", separatorWidth)
            .attr("stroke-dasharray", dashArray);
    });
}

function drawHiddenState(g, matrix, labels, width, height, separators, layerIdx, deletedPoints = []) {
    const numTokens = matrix.length; 
    const hiddenDim = matrix[0].length; 
    const cellW = width / hiddenDim;
    const cellH = height / numTokens;
    const colorScale = d3.scaleSequential(CONFIG.style.colors.hidden).domain([-1, 1]);
    
    // 1. Draw Data Rects
    matrix.forEach((tokenRow, i) => {
        g.append("text")
            .attr("x", -10).attr("y", i * cellH + cellH / 2)
            .attr("text-anchor", "end").attr("alignment-baseline", "middle")
            .style("fill", "#fafafa").style("font-size", "12px").text(labels[i]);
        
        tokenRow.forEach((value, j) => {
           
            g.append("rect")
                .attr("x", j * cellW).attr("y", i * cellH)
                .attr("width", cellW).attr("height", cellH)
                .style("fill", colorScale(value))
                .on("mouseover", (event) => {
                    // Tooltips only work if toggle is OFF
                    if (isPerturbationMode) return; 
                    d3.select(event.currentTarget).style("stroke", "white").style("stroke-width", 1);
                    showTooltip(event, `Token: ${labels[i]}`, `Dim: ${j}`, value);
                })
                .on("mouseout", (event) =>{
                    d3.select(event.currentTarget).style("stroke", "none");
                    hideTooltip();
                });
        });
    });

    // 2. Define Brush
    const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on("end", (event) => {
            // 1. GUARD: If there's no sourceEvent, this was a programmatic call (like brush.move)
            // We must return immediately to avoid the "target of undefined" error.
            if (!event.sourceEvent) return; 

            // 2. CHECK: If no selection exists or we aren't in ablation mode, just clear and exit
            if (!event.selection || !isPerturbationMode) {
                d3.select(event.sourceEvent.target.parentNode).call(brush.move, null);
                return;
            }

            const [[x0, y0], [x1, y1]] = event.selection;
            const startDim = Math.floor(x0 / cellW);
            const endDim = Math.floor(x1 / cellW);
            const startToken = Math.floor(y0 / cellH);
            const endToken = Math.floor(y1 / cellH);
            
            const points = [];
            for (let t = startToken; t <= endToken; t++) {
                for (let d = startDim; d <= endDim; d++) {
                    points.push({ layer: layerIdx, token: t, dim: d });
                }
            }

            // 3. Clear the visual brush box
            // We use the guard above to make sure THIS call doesn't cause a loop/crash
            d3.select(event.sourceEvent.target.parentNode).call(brush.move, null);
            
            // 4. Send to Streamlit
            sendToStreamlit({ type: "ABLATE_COORDINATES", points: points });
        });

    // 3. Initialize Brush Group (Locked by default)
    const brushG = g.append("g")
        .attr("class", "brush")
        .call(brush);

    // Ensure the overlay is unreachable until the toggle is switched on
    brushG.style("pointer-events", "none");
    brushG.select(".overlay").style("pointer-events", "none");

    (separators || []).forEach(pos => {
        const lineY = pos * cellH;
        g.append("line").attr("x1", 0).attr("y1", lineY).attr("x2", width).attr("y2", lineY)
            .attr("stroke", "#ffffff").attr("stroke-width", 2).attr("stroke-dasharray", "4,2");
    });
}

function drawNextTokenPredictions(g, labels, yOffset, xOffset, predictionData, cellSize) {
    if (!predictionData || !Array.isArray(predictionData)) return;
    const plotHeight = 200;
    const plotWidth = Math.max(cellSize * 0.9, 60);
    const gap = 40;
    const predG = g.append("g").attr("transform", `translate(${xOffset}, ${yOffset + 100})`);

    predG.append("text")
        .attr("y", -60).style("fill", "#ffffff").style("font-size", "30px").style("font-weight", "bold")
        .text("Next Token Probability Distributions");

    predictionData.forEach((topK, i) => {
        const xPos = i * (plotWidth + gap);
        const cellG = predG.append("g").attr("transform", `translate(${xPos}, 0)`);
        const yScale = d3.scaleLinear().domain([0, 1]).range([plotHeight, 0]);
        const xScale = d3.scaleBand().domain(topK.map(d => d.label)).range([0, plotWidth]).padding(0.2);

        if (i === 0) {
            cellG.append("g").call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".1f")))
                .selectAll("text").style("fill", "#888").style("font-size", "14px");
        }

        cellG.selectAll(".bar").data(topK).enter().append("rect")
            .attr("x", d => xScale(d.label)).attr("y", d => yScale(d.prob))
            .attr("width", xScale.bandwidth()).attr("height", d => plotHeight - yScale(d.prob))
            .attr("fill", (d, idx) => d3.interpolateViridis(1 - idx / 5))
            .on("mouseover", (event, d) => showTooltip(event, "Char", d.label, d.prob))
            .on("mouseout", hideTooltip);

        cellG.append("g").attr("transform", `translate(0, ${plotHeight})`).call(d3.axisBottom(xScale))
            .selectAll("text").style("fill", "#fafafa").style("font-size", "18px");

        cellG.append("text").attr("x", plotWidth / 2).attr("y", -20).attr("text-anchor", "middle")
            .style("fill", "#FFA500").style("font-size", "20px").style("font-style", "italic").text(`Gen Token #${i+1}`);
    });
}

function drawConnection(parent, points, type = "flow") {
    const isStandard = type === "flow";
    const color = isStandard ? CONFIG.style.colors.connection : CONFIG.style.colors.highlight;
    const marker = isStandard ? "url(#flow-arrow)" : "url(#decoder-arrow)";

    if (Array.isArray(points[0])) {
        return parent.append("path").attr("d", d3.line()(points)).attr("fill", "none")
            .attr("stroke", color).attr("stroke-width", 3).attr("stroke-dasharray", "8,4").attr("marker-end", marker);
    } 
    return parent.append("line").attr("x1", points.x1).attr("y1", points.y1).attr("x2", points.x2).attr("y2", points.y2)
        .attr("stroke", color).attr("stroke-width", 3).attr("stroke-dasharray", "8,4").attr("marker-end", marker);
}

// --- Main Render Logic ---
function renderComputationalGraph(attentionWeights, hiddenStates, labels, separators, decoderLayerIndex = 2, predictionData = null, deletedList = []) {
    const { margin, verticalGap, gutterWidth } = CONFIG.layout;
    const { encoderBlock, decoder, predictions } = CONFIG.nodes;
    const { colors } = CONFIG.style;

    const encoderStackHeight = (attentionWeights.length * encoderBlock.heatmapSize) + ((attentionWeights.length - 1) * verticalGap);
    const xOffsetHidden = margin.left + encoderBlock.heatmapSize + gutterWidth;
    const yOffsetHidden = 250;
    const inputBlockHeight = encoderBlock.heatmapSize + 50

    const totalHeight = margin.top + inputBlockHeight + yOffsetHidden + encoderStackHeight + decoder.topMargin + margin.bottom + encoderBlock.heatmapSize;
    const totalWidth = margin.left + encoderBlock.heatmapSize + gutterWidth + encoderBlock.hiddenWidth + margin.right;
    
    const container = d3.select("#chart-container");
    container.selectAll("*").remove();
    const svg = container.append("svg")
        .attr("width", totalWidth)
        .attr("height", totalHeight);

    // Initial attachment of zoom
    svg.call(zoom);

    // ADD DOUBLE CLICK TO RESET
    svg.on("dblclick.zoom", function(event) {
        svg.transition()
            .duration(750)
            .call(zoom.transform, d3.zoomIdentity); // Resets to original scale and 0,0
    });

    // If we start in ablation mode, we need to disable zoom immediately
    if (isPerturbationMode) {
        svg.on(".zoom", null); 
    }
    
    const mainGroup = svg.append("g");

    const defs = svg.append("defs");
    const addArrowhead = (id, color) => {
        defs.append("marker").attr("id", id).attr("viewBox", "0 -5 10 10").attr("refX", 9).attr("refY", 0)
            .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto")
            .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", color);
    };
    addArrowhead("flow-arrow", colors.connection);
    addArrowhead("decoder-arrow", colors.highlight);

    

    // Inputs (before encoder block)
    mainGroup.append("text").attr("x", xOffsetHidden).attr("y", margin.top - 15)
        .style("fill", colors.textDim).style("font-size", "16px").text(`Input Embeddings`);

    const hiddenG = mainGroup.append("g").attr("transform", `translate(${xOffsetHidden}, ${margin.top})`);


    drawHiddenState(hiddenG, hiddenStates[0], labels, encoderBlock.hiddenWidth, encoderBlock.heatmapSize, separators, 0, []);
    drawConnection(mainGroup, {
        x1: (encoderBlock.hiddenWidth / 2) + xOffsetHidden, y1: margin.top + inputBlockHeight,
        x2: (encoderBlock.hiddenWidth / 2) + xOffsetHidden, y2: margin.top + inputBlockHeight + yOffsetHidden + 70
    }, "flow");

    
    // Encoder
    mainGroup.append("text").attr("x", margin.left).attr("y", margin.top + inputBlockHeight)
        .style("fill", colors.textMain).style("font-size", "22px").style("font-weight", "bold").text("ENCODER");

    mainGroup.append("rect").attr("x", margin.left - 50).attr("y", margin.top + inputBlockHeight + 20)
        .attr("width", encoderBlock.heatmapSize + gutterWidth + encoderBlock.hiddenWidth + 100)
        .attr("height", encoderStackHeight + yOffsetHidden + 100).attr("fill", colors.bgBox).attr("stroke", "#444").attr("rx", encoderBlock.borderRadius);

    drawAblationToggle(mainGroup, xOffsetHidden + encoderBlock.hiddenWidth - CONFIG.style.ui.toggleWidth, margin.top - 80);

    // Encoder Layers
    attentionWeights.forEach((matrices, idx) => {
        const yOffset = margin.top + ((idx + 1) * (encoderBlock.heatmapSize + verticalGap));
        
        mainGroup.append("text").attr("x", margin.left).attr("y", yOffset - 15)
            .style("fill", colors.textDim).style("font-size", "16px")
            .text(`Layer ${idx + 1} Attention (${matrices.length === 1 ? 'Averaged' : 'Individual Heads'})`);

        const layerGroup = mainGroup.append("g").attr("transform", `translate(${margin.left}, ${yOffset})`);

        if (matrices.length === 1) {
            // Case 1: Averaged View (Single Matrix)
            const matrix = matrices[0];
            const cellSize = encoderBlock.heatmapSize / matrix.length;
            drawHeatmap(layerGroup, matrix, labels, cellSize, true, separators);
        } else {
            // --- CASE 2: INDIVIDUAL HEADS VIEW (3x3 Grid with Titles & Padding) ---
            const gridCols = 3;
            const padding = encoderBlock.gridPadding;
            const titleSpace = encoderBlock.titleSpace;
            
            // Calculate the size for each individual head "slot" including its title
            // (Total Width - total padding) / columns
            const slotSize = (encoderBlock.heatmapSize - (padding * (gridCols - 1))) / gridCols;
            
            // The actual heatmap area is the slotSize minus the vertical space for the title
            const actualHeatmapHeight = slotSize - titleSpace;

            matrices.forEach((matrix, headIdx) => {
                const row = Math.floor(headIdx / gridCols);
                const col = headIdx % gridCols;
                
                // Positioning accounts for the slotSize + padding
                const headX = col * (slotSize + padding);
                const headY = row * (slotSize + padding);
                
                const headGroup = layerGroup.append("g")
                    .attr("transform", `translate(${headX}, ${headY})`);

                // 1. Add Head Title
                headGroup.append("text")
                    .attr("x", slotSize / 2)
                    .attr("y", titleSpace - 5) // Position in the reserved space
                    .attr("text-anchor", "middle")
                    .style("fill", colors.textDim)
                    .style("font-size", "10px")
                    .style("font-weight", "bold")
                    .text(`Head ${headIdx+1}`);

                // 2. Create a nested group for the heatmap to sit below the title
                const heatmapArea = headGroup.append("g")
                    .attr("transform", `translate(0, ${titleSpace})`);

                // 3. Draw the Heatmap
                // The cellSize is based on the actualHeatmapHeight
                const cellSize = actualHeatmapHeight / matrix.length;
                
                drawHeatmap(heatmapArea, matrix, labels, cellSize, true, separators, "small");
            });
        }

        // Draw horizontal connections to attention:
        const centerX = (encoderBlock.hiddenWidth / 2) + xOffsetHidden;

        drawConnection(mainGroup, {
            x1: centerX - 20, y1: yOffset + encoderBlock.heatmapSize / 2 - 30,
            x2: margin.left + encoderBlock.heatmapSize + 20, y2: yOffset + encoderBlock.heatmapSize / 2 - 30
        }, "flow");

        drawConnection(mainGroup, {
            x1: margin.left + encoderBlock.heatmapSize + 20, y1: yOffset + encoderBlock.heatmapSize / 2,
            x2: centerX - 20, y2: yOffset + encoderBlock.heatmapSize / 2
        }, "flow");

        // Draw Hidden States
        mainGroup.append("text").attr("x", xOffsetHidden).attr("y", yOffset + yOffsetHidden - 15)
            .style("fill", colors.textDim).style("font-size", "16px").text(`Layer ${idx + 1} Hidden States`);
        // visualize perturbed hidden states:
        const layerDeletedPoints = deletedList.filter(p => p.layer === idx + 1);
        const hiddenG = mainGroup.append("g").attr("transform", `translate(${xOffsetHidden}, ${yOffset + yOffsetHidden})`);
        drawHiddenState(hiddenG, hiddenStates[idx + 1], labels, encoderBlock.hiddenWidth, encoderBlock.heatmapSize, separators, idx, layerDeletedPoints);
        
        // Draw vertical connections between hidden states:
        if (idx < attentionWeights.length - 1) {
            
            drawConnection(mainGroup, {
                x1: centerX, y1: yOffset + encoderBlock.heatmapSize + yOffsetHidden + 30,
                x2: centerX, y2: yOffset + encoderBlock.heatmapSize + yOffsetHidden + verticalGap - 30
            }, "flow");
        }
    });

    const encoderRightEdge = xOffsetHidden + encoderBlock.hiddenWidth;
    const decoderX = margin.left + (encoderBlock.heatmapSize + gutterWidth + encoderBlock.hiddenWidth) / 2 - (decoder.width / 2);
    const decoderY = margin.top + inputBlockHeight + encoderStackHeight + decoder.topMargin + yOffsetHidden;
    const selectedY = margin.top + inputBlockHeight + (decoderLayerIndex * (encoderBlock.heatmapSize + verticalGap)) + (encoderBlock.heatmapSize / 2) + yOffsetHidden;
    const bottomEdgeEncoder = margin.top + inputBlockHeight + encoderStackHeight + yOffsetHidden + 120
    
    const route = [
        [encoderRightEdge + 10, selectedY], 
        [encoderRightEdge + 80, selectedY],
        [encoderRightEdge + 80, bottomEdgeEncoder + 30],
        [decoderX + decoder.width/2, bottomEdgeEncoder + 30], 
        [decoderX + decoder.width/2, decoderY - 10]
    ];

    drawConnection(mainGroup, route, "decoder");

    const decoderG = mainGroup.append("g").attr("transform", `translate(${decoderX}, ${decoderY})`);
    decoderG.append("rect").attr("width", decoder.width).attr("height", decoder.height).attr("rx", decoder.borderRadius)
        .attr("fill", "#111").attr("stroke", colors.connection).attr("stroke-width", 3);
    decoderG.append("text").attr("x", decoder.width/2).attr("y", decoder.height/2 + 8).attr("text-anchor", "middle")
        .style("fill", "white").style("font-weight", "bold").style("font-size", "22px").text("DECODER");
    
    const predY = decoderY + decoder.height + predictions.topMargin;
    console.log(predY);
    drawConnection(mainGroup, { x1: decoderX + decoder.width/2, y1: decoderY + decoder.height, x2: decoderX + decoder.width/2, y2: predY - 20 }, "decoder");
    drawNextTokenPredictions(mainGroup, labels, predY, margin.left, predictionData, encoderBlock.heatmapSize);
    setFrameHeight(totalHeight)
}

function onRender(data) {
    const args = data.args;
    d3.select("#chart-container").selectAll("*").remove();

    if (args.view === "full_graph") {
        renderComputationalGraph(
            args.attention_weights, 
            args.hidden_states, 
            args.labels,
            args.separators,
            args.decoder_layer_index,
            args.prediction_data,
            args.deleted_list || [] // Default to empty if not passed
        );
    }
}

window.addEventListener("message", (event) => {
  if (event.data.type === "streamlit:render") onRender(event.data);
});

window.addEventListener("load", () => {
    // Give the browser 50ms to ensure the iframe is fully recognized by Streamlit
    setTimeout(() => {
        sendReady();
    }, 50);
});