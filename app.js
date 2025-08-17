// Real-Time Neuromorphic Spectrum Intelligence Simulator
// Full-Stack Implementation with Live Updates and Export Functionality

class RealTimeNeuromorphicSimulator {
    constructor() {
        this.isRunning = false;
        this.isPaused = false;
        this.currentTime = 0;
        this.totalDuration = 300;
        this.updateInterval = null;
        this.charts = {};
        this.currentChart = 'performance';
        this.updateRate = 0;
        this.lastUpdate = Date.now();
        
        // Real-time simulation state
        this.state = {
            config: {
                numNodes: 1000,
                neuromorphicType: 'snn',
                energySources: ['triboelectric', 'rf-harvesting'],
                adversarialRatio: 0.3,
                attackTypes: ['jamming', 'spoofing'],
                learningRate: 0.01,
                simulationDuration: 300
            },
            metrics: {
                performanceScore: 0,
                learningAccuracy: 0.5,
                spectrumUtilization: 0,
                energyEfficiency: 23.6,
                adversarialImpact: 0,
                activeNodes: 1000
            },
            timeline: [],
            nodeStatus: [],
            previousMetrics: {}
        };

        // WebSocket simulation
        this.websocket = {
            connected: true,
            send: (data) => this.handleWebSocketMessage(data),
            onmessage: null,
            onerror: null,
            onopen: null,
            onclose: null
        };

        this.init();
    }

    init() {
        console.log('🚀 Initializing Real-Time Neuromorphic Simulator...');
        this.setupEventListeners();
        this.initializeCharts();
        this.generateInitialNodeStatus();
        this.updateDashboard();
        this.startHeartbeat();
        console.log('✅ Real-Time Simulator Ready');
    }

    setupEventListeners() {
        // Real-time parameter controls
        document.getElementById('num-nodes').addEventListener('input', (e) => {
            this.updateParameter('numNodes', parseInt(e.target.value));
            document.getElementById('nodes-value').textContent = e.target.value;
        });

        document.getElementById('sim-duration').addEventListener('input', (e) => {
            this.updateParameter('simulationDuration', parseInt(e.target.value));
            document.getElementById('duration-value').textContent = e.target.value + 's';
            document.getElementById('total-duration').textContent = e.target.value + 's';
            this.totalDuration = parseInt(e.target.value);
        });

        document.getElementById('neuromorphic-type').addEventListener('change', (e) => {
            this.updateParameter('neuromorphicType', e.target.value);
        });

        document.getElementById('learning-rate').addEventListener('input', (e) => {
            this.updateParameter('learningRate', parseFloat(e.target.value));
            document.getElementById('learning-rate-value').textContent = e.target.value;
        });

        document.getElementById('adversarial-ratio').addEventListener('input', (e) => {
            this.updateParameter('adversarialRatio', parseInt(e.target.value) / 100);
            document.getElementById('adversarial-value').textContent = e.target.value + '%';
        });

        // Energy source checkboxes
        ['triboelectric', 'rf-harvesting', 'ionic-wind'].forEach(source => {
            const checkbox = document.getElementById(source);
            if (checkbox) {
                checkbox.addEventListener('change', () => {
                    this.updateEnergySources();
                });
            }
        });

        // Attack type checkboxes
        ['jamming', 'spoofing', 'byzantine'].forEach(attack => {
            const checkbox = document.getElementById(attack);
            if (checkbox) {
                checkbox.addEventListener('change', () => {
                    this.updateAttackTypes();
                });
            }
        });

        // Simulation controls
        document.getElementById('start-simulation').addEventListener('click', () => {
            this.isRunning ? this.stopSimulation() : this.startSimulation();
        });

        document.getElementById('pause-simulation').addEventListener('click', () => {
            this.pauseSimulation();
        });

        document.getElementById('reset-simulation').addEventListener('click', () => {
            this.resetSimulation();
        });

        // Chart navigation - Fixed implementation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const chartType = e.target.dataset.chart;
                console.log(`🎯 Switching to chart: ${chartType}`);
                this.switchChart(chartType);
            });
        });

        // Node filtering
        document.getElementById('node-filter').addEventListener('change', (e) => {
            this.filterNodes(e.target.value);
        });

        // Export functions
        document.getElementById('export-json').addEventListener('click', () => this.exportData('json'));
        document.getElementById('export-csv').addEventListener('click', () => this.exportData('csv'));
        document.getElementById('export-nodes').addEventListener('click', () => this.exportData('nodes'));
        document.getElementById('export-config').addEventListener('click', () => this.exportData('config'));
    }

    updateParameter(key, value) {
        console.log(`🔧 Parameter Update: ${key} = ${value}`);
        this.state.config[key] = value;
        
        // Send WebSocket-like message
        this.websocket.send({
            type: 'parameter_changed',
            parameter: key,
            value: value,
            timestamp: Date.now()
        });

        // Apply immediate effect if simulation is running
        if (this.isRunning) {
            this.applyParameterChange(key, value);
        }
    }

    updateEnergySources() {
        const sources = [];
        ['triboelectric', 'rf-harvesting', 'ionic-wind'].forEach(source => {
            const checkbox = document.getElementById(source);
            if (checkbox && checkbox.checked) {
                sources.push(source);
            }
        });
        this.updateParameter('energySources', sources);
    }

    updateAttackTypes() {
        const attacks = [];
        ['jamming', 'spoofing', 'byzantine'].forEach(attack => {
            const checkbox = document.getElementById(attack);
            if (checkbox && checkbox.checked) {
                attacks.push(attack);
            }
        });
        this.updateParameter('attackTypes', attacks);
    }

    applyParameterChange(key, value) {
        switch (key) {
            case 'numNodes':
                this.state.metrics.activeNodes = value;
                this.generateNodeStatus();
                break;
            case 'adversarialRatio':
                this.updateAdversarialNodes();
                break;
            case 'learningRate':
                // Adjust learning convergence speed
                this.adjustLearningRate(value);
                break;
            case 'energySources':
                this.updateEnergyMetrics();
                break;
        }
        
        // Update dashboard immediately
        this.updateDashboard();
        this.updateCharts();
    }

    startSimulation() {
        console.log('🎯 Starting Real-Time Simulation...');
        this.isRunning = true;
        this.isPaused = false;

        // Update UI
        const startBtn = document.getElementById('start-simulation');
        const pauseBtn = document.getElementById('pause-simulation');
        const spinner = document.getElementById('sim-spinner');
        
        startBtn.textContent = 'Stop Simulation';
        startBtn.classList.add('running');
        pauseBtn.disabled = false;
        spinner.classList.remove('hidden');
        
        document.getElementById('sim-status').textContent = 'Running';
        document.getElementById('sim-status').className = 'status--success';
        document.getElementById('ws-status').textContent = 'Streaming';
        document.getElementById('ws-status').className = 'status--warning';

        // Start real-time updates
        this.updateInterval = setInterval(() => {
            this.performRealtimeUpdate();
        }, 1000); // 1Hz updates

        // Start progress tracking
        this.startProgressTracking();
        
        console.log('✅ Real-Time Simulation Started');
    }

    stopSimulation() {
        console.log('🛑 Stopping Simulation...');
        this.isRunning = false;
        this.isPaused = false;

        // Clear intervals
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }

        // Update UI
        const startBtn = document.getElementById('start-simulation');
        const pauseBtn = document.getElementById('pause-simulation');
        const spinner = document.getElementById('sim-spinner');
        
        startBtn.textContent = 'Start Simulation';
        startBtn.classList.remove('running');
        pauseBtn.disabled = true;
        spinner.classList.add('hidden');
        
        document.getElementById('sim-status').textContent = 'Stopped';
        document.getElementById('sim-status').className = 'status--error';
        document.getElementById('ws-status').textContent = 'Connected';
        document.getElementById('ws-status').className = 'status--success';
        
        console.log('✅ Simulation Stopped');
    }

    pauseSimulation() {
        if (this.isPaused) {
            console.log('▶️ Resuming Simulation...');
            this.isPaused = false;
            document.getElementById('pause-simulation').textContent = 'Pause';
            this.updateInterval = setInterval(() => {
                this.performRealtimeUpdate();
            }, 1000);
        } else {
            console.log('⏸️ Pausing Simulation...');
            this.isPaused = true;
            document.getElementById('pause-simulation').textContent = 'Resume';
            if (this.updateInterval) {
                clearInterval(this.updateInterval);
                this.updateInterval = null;
            }
        }
    }

    resetSimulation() {
        console.log('🔄 Resetting Simulation...');
        this.stopSimulation();
        
        // Reset state
        this.currentTime = 0;
        this.state.timeline = [];
        this.state.metrics = {
            performanceScore: 0,
            learningAccuracy: 0.5,
            spectrumUtilization: 0,
            energyEfficiency: 23.6,
            adversarialImpact: 0,
            activeNodes: this.state.config.numNodes
        };

        // Reset UI
        document.getElementById('simulation-time').textContent = '0.0s';
        document.getElementById('progress-fill').style.width = '0%';
        
        // Reset charts
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.data.datasets.forEach(dataset => {
                    dataset.data = [];
                });
                chart.data.labels = [];
                chart.update();
            }
        });

        this.generateNodeStatus();
        this.updateDashboard();
        
        console.log('✅ Simulation Reset Complete');
    }

    performRealtimeUpdate() {
        if (!this.isRunning || this.isPaused) return;

        this.currentTime += 1;
        const progress = (this.currentTime / this.totalDuration) * 100;

        // Update progress UI
        document.getElementById('simulation-time').textContent = this.currentTime.toFixed(1) + 's';
        document.getElementById('progress-fill').style.width = Math.min(progress, 100) + '%';

        // Generate new data point
        const dataPoint = this.generateDataPoint();
        this.state.timeline.push(dataPoint);

        // Update metrics with real-time calculations
        this.updateMetricsFromDataPoint(dataPoint);
        
        // Update charts with new data
        this.updateChartsWithNewData(dataPoint);
        
        // Update dashboard with animations
        this.updateDashboardAnimated();
        
        // Update node status
        this.updateNodeStatus();
        
        // Update performance indicators
        this.updateUpdateRate();

        // Auto-stop when duration reached
        if (this.currentTime >= this.totalDuration) {
            this.stopSimulation();
            console.log('🏁 Simulation completed successfully');
        }

        // Send WebSocket-like message
        this.websocket.send({
            type: 'simulation_data',
            data: dataPoint,
            timestamp: Date.now()
        });
    }

    generateDataPoint() {
        const t = this.currentTime;
        const config = this.state.config;
        
        // Realistic data generation based on current parameters
        const baseNoise = () => (Math.random() - 0.5) * 0.1;
        const cyclicPattern = (amplitude, period, phase = 0) => 
            amplitude * Math.sin(2 * Math.PI * t / period + phase);
        
        // Performance score calculation
        const learningProgress = Math.min(0.9, 0.5 + (t / this.totalDuration) * 0.4 * config.learningRate * 10);
        const adversarialPenalty = config.adversarialRatio * 0.3;
        const energyBonus = config.energySources.length * 0.1;
        const performanceScore = Math.max(0, Math.min(100, 
            (learningProgress - adversarialPenalty + energyBonus) * 100 + baseNoise() * 5
        ));

        // Learning accuracy with convergence
        const learningAccuracy = Math.min(0.95, 
            0.5 + (1 - Math.exp(-t * config.learningRate * 0.1)) * 0.4 + baseNoise() * 0.05
        );

        // Spectrum utilization with realistic patterns
        const spectrumUtilization = Math.max(0.1, Math.min(0.9,
            0.3 + cyclicPattern(0.2, 60) + cyclicPattern(0.1, 20, Math.PI/4) + baseNoise() * 0.1
        ));

        // Energy efficiency based on node count and energy sources
        const nodeEfficiencyFactor = Math.log(config.numNodes) / Math.log(1000);
        const sourceEfficiencyFactor = config.energySources.length * 0.2;
        const energyEfficiency = 23.6 * (1 + sourceEfficiencyFactor) * nodeEfficiencyFactor + baseNoise() * 2;

        // Adversarial impact
        const adversarialImpact = config.adversarialRatio * 
            (0.5 + cyclicPattern(0.3, 45, Math.PI/3) + baseNoise() * 0.2);

        // Throughput calculation
        const baselineThrough = 40e6; // 40 Mbps baseline
        const throughput = baselineThrough * spectrumUtilization * 
            (1 + energyBonus) * (1 - adversarialPenalty) + baseNoise() * 5e6;

        return {
            timestamp: t,
            performanceScore,
            learningAccuracy,
            spectrumUtilization,
            energyEfficiency,
            adversarialImpact,
            throughput,
            activeNodes: config.numNodes * (0.9 + baseNoise() * 0.1),
            spikeRate: 8.5 + cyclicPattern(2, 30) + baseNoise() * 1,
            energyHarvested: (2.5 + cyclicPattern(1, 40)) * config.energySources.length * 1e-6
        };
    }

    updateMetricsFromDataPoint(dataPoint) {
        // Store previous values for change calculation
        this.state.previousMetrics = { ...this.state.metrics };
        
        // Update current metrics
        this.state.metrics = {
            performanceScore: dataPoint.performanceScore,
            learningAccuracy: dataPoint.learningAccuracy,
            spectrumUtilization: dataPoint.spectrumUtilization,
            energyEfficiency: dataPoint.energyEfficiency,
            adversarialImpact: dataPoint.adversarialImpact,
            activeNodes: Math.round(dataPoint.activeNodes)
        };
    }

    updateDashboardAnimated() {
        const metrics = this.state.metrics;
        const previous = this.state.previousMetrics;
        
        // Performance Score
        const perfCard = document.getElementById('performance-card');
        const perfValue = document.getElementById('performance-score');
        const perfChange = document.getElementById('performance-change');
        const perfStatus = document.getElementById('performance-status');
        
        perfValue.textContent = metrics.performanceScore.toFixed(1);
        if (previous.performanceScore) {
            const change = metrics.performanceScore - previous.performanceScore;
            perfChange.textContent = (change > 0 ? '+' : '') + change.toFixed(1);
            perfChange.className = `metric-card__change ${change > 0 ? 'positive' : change < 0 ? 'negative' : ''}`;
        }
        perfCard.classList.add('updating');
        perfStatus.className = `status-indicator ${this.isRunning ? 'active' : ''}`;
        setTimeout(() => perfCard.classList.remove('updating'), 500);

        // Energy Efficiency
        const energyValue = document.getElementById('energy-efficiency');
        const energyChange = document.getElementById('energy-change');
        const energyStatus = document.getElementById('energy-status');
        
        energyValue.textContent = metrics.energyEfficiency.toFixed(1) + ' pJ';
        if (previous.energyEfficiency) {
            const change = metrics.energyEfficiency - previous.energyEfficiency;
            energyChange.textContent = (change > 0 ? '+' : '') + change.toFixed(1);
            energyChange.className = `metric-card__change ${change > 0 ? 'positive' : change < 0 ? 'negative' : ''}`;
        }
        energyStatus.className = `status-indicator ${this.isRunning ? 'active' : ''}`;

        // Spectrum Utilization
        const spectrumValue = document.getElementById('spectrum-utilization');
        const spectrumChange = document.getElementById('spectrum-change');
        const spectrumStatus = document.getElementById('spectrum-status');
        
        spectrumValue.textContent = (metrics.spectrumUtilization * 100).toFixed(1) + '%';
        if (previous.spectrumUtilization) {
            const change = (metrics.spectrumUtilization - previous.spectrumUtilization) * 100;
            spectrumChange.textContent = (change > 0 ? '+' : '') + change.toFixed(1) + '%';
            spectrumChange.className = `metric-card__change ${change > 0 ? 'positive' : change < 0 ? 'negative' : ''}`;
        }
        spectrumStatus.className = `status-indicator ${this.isRunning ? 'active' : ''}`;

        // Active Nodes
        const nodesValue = document.getElementById('active-nodes');
        const nodesChange = document.getElementById('nodes-change');
        const nodesStatus = document.getElementById('nodes-status');
        
        nodesValue.textContent = metrics.activeNodes.toString();
        if (previous.activeNodes) {
            const change = metrics.activeNodes - previous.activeNodes;
            nodesChange.textContent = (change > 0 ? '+' : '') + change.toString();
            nodesChange.className = `metric-card__change ${change > 0 ? 'positive' : change < 0 ? 'negative' : ''}`;
        }
        nodesStatus.className = `status-indicator ${this.isRunning ? 'active' : ''}`;
    }

    initializeCharts() {
        console.log('📊 Initializing Real-Time Charts...');
        
        this.createPerformanceChart();
        this.createLearningChart();
        this.createSpectrumChart();
        this.createEnergyChart();
        
        console.log('✅ All charts initialized');
    }

    createPerformanceChart() {
        const ctx = document.getElementById('performance-canvas').getContext('2d');
        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Performance Score',
                    data: [],
                    borderColor: '#1FB8CD',
                    backgroundColor: 'rgba(31, 184, 205, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 1,
                    pointHoverRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 750 },
                plugins: {
                    title: {
                        display: true,
                        text: 'Real-Time Performance Score',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Time (seconds)' },
                        type: 'linear'
                    },
                    y: {
                        title: { display: true, text: 'Performance Score' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    createLearningChart() {
        const ctx = document.getElementById('learning-canvas').getContext('2d');
        this.charts.learning = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Learning Accuracy',
                    data: [],
                    borderColor: '#FFC185',
                    backgroundColor: 'rgba(255, 193, 133, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 750 },
                plugins: {
                    title: {
                        display: true,
                        text: 'Federated Learning Accuracy',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Time (seconds)' } },
                    y: { 
                        title: { display: true, text: 'Accuracy' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    createSpectrumChart() {
        const ctx = document.getElementById('spectrum-canvas').getContext('2d');
        this.charts.spectrum = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Spectrum Utilization',
                    data: [],
                    borderColor: '#B4413C',
                    backgroundColor: 'rgba(180, 65, 60, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 750 },
                plugins: {
                    title: {
                        display: true,
                        text: 'Spectrum Utilization Over Time',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Time (seconds)' } },
                    y: { 
                        title: { display: true, text: 'Utilization' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    createEnergyChart() {
        const ctx = document.getElementById('energy-canvas').getContext('2d');
        this.charts.energy = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Energy Harvested (µJ)',
                        data: [],
                        borderColor: '#ECEBD5',
                        backgroundColor: 'rgba(236, 235, 213, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Spike Rate',
                        data: [],
                        borderColor: '#5D878F',
                        backgroundColor: 'rgba(93, 135, 143, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 750 },
                plugins: {
                    title: {
                        display: true,
                        text: 'Energy Harvesting & Neural Activity',
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Time (seconds)' } },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: 'Energy (µJ)' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: 'Spike Rate' },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
    }

    updateChartsWithNewData(dataPoint) {
        Object.entries(this.charts).forEach(([key, chart]) => {
            if (!chart) return;

            // Add timestamp to labels
            chart.data.labels.push(dataPoint.timestamp);

            // Update datasets based on chart type
            switch (key) {
                case 'performance':
                    chart.data.datasets[0].data.push(dataPoint.performanceScore);
                    break;
                case 'learning':
                    chart.data.datasets[0].data.push(dataPoint.learningAccuracy);
                    break;
                case 'spectrum':
                    chart.data.datasets[0].data.push(dataPoint.spectrumUtilization);
                    break;
                case 'energy':
                    chart.data.datasets[0].data.push(dataPoint.energyHarvested * 1e6);
                    chart.data.datasets[1].data.push(dataPoint.spikeRate);
                    break;
            }

            // Keep only last 60 data points for performance
            const maxPoints = 60;
            if (chart.data.labels.length > maxPoints) {
                chart.data.labels = chart.data.labels.slice(-maxPoints);
                chart.data.datasets.forEach(dataset => {
                    dataset.data = dataset.data.slice(-maxPoints);
                });
            }

            // Update chart with smooth animation
            chart.update('none');
        });
    }

    switchChart(chartType) {
        console.log(`📊 Switching to chart: ${chartType}`);
        
        // Hide all chart containers
        const containers = ['performance-chart', 'learning-chart', 'spectrum-chart', 'energy-chart'];
        containers.forEach(containerId => {
            const container = document.getElementById(containerId);
            if (container) {
                container.classList.add('hidden');
                console.log(`📊 Hidden: ${containerId}`);
            }
        });
        
        // Show selected chart container
        const targetContainer = document.getElementById(`${chartType}-chart`);
        if (targetContainer) {
            targetContainer.classList.remove('hidden');
            console.log(`📊 Shown: ${chartType}-chart`);
        } else {
            console.error(`❌ Container not found: ${chartType}-chart`);
        }
        
        // Update tab button states
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeTab = document.querySelector(`[data-chart="${chartType}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
            console.log(`📊 Activated tab: ${chartType}`);
        } else {
            console.error(`❌ Tab button not found for: ${chartType}`);
        }
        
        this.currentChart = chartType;
        
        // Resize chart if it exists
        const chart = this.charts[chartType];
        if (chart) {
            setTimeout(() => {
                chart.resize();
                console.log(`📊 Resized chart: ${chartType}`);
            }, 100);
        }
        
        console.log(`✅ Chart switched successfully to: ${chartType}`);
    }

    generateInitialNodeStatus() {
        this.state.nodeStatus = [];
        for (let i = 0; i < this.state.config.numNodes; i++) {
            const isAdversarial = Math.random() < this.state.config.adversarialRatio;
            this.state.nodeStatus.push({
                id: i,
                status: isAdversarial ? 'adversarial' : 'active',
                lastUpdate: Date.now()
            });
        }
        this.renderNodeGrid();
    }

    generateNodeStatus() {
        // Update node count if changed
        const targetCount = this.state.config.numNodes;
        const currentCount = this.state.nodeStatus.length;
        
        if (targetCount > currentCount) {
            // Add nodes
            for (let i = currentCount; i < targetCount; i++) {
                this.state.nodeStatus.push({
                    id: i,
                    status: 'active',
                    lastUpdate: Date.now()
                });
            }
        } else if (targetCount < currentCount) {
            // Remove nodes
            this.state.nodeStatus = this.state.nodeStatus.slice(0, targetCount);
        }
        
        this.renderNodeGrid();
    }

    updateNodeStatus() {
        if (!this.isRunning) return;
        
        // Randomly update some node statuses
        const updateCount = Math.floor(this.state.nodeStatus.length * 0.05); // 5% of nodes
        for (let i = 0; i < updateCount; i++) {
            const nodeIndex = Math.floor(Math.random() * this.state.nodeStatus.length);
            const node = this.state.nodeStatus[nodeIndex];
            
            if (node.status === 'adversarial') continue; // Don't change adversarial nodes
            
            // Randomly make nodes inactive/active
            node.status = Math.random() < 0.1 ? 'inactive' : 'active';
            node.lastUpdate = Date.now();
        }
        
        this.updateNodeCounts();
        this.renderNodeGrid();
    }

    updateAdversarialNodes() {
        const adversarialCount = Math.floor(this.state.config.numNodes * this.state.config.adversarialRatio);
        
        // Reset all to active first
        this.state.nodeStatus.forEach(node => {
            if (node.status === 'adversarial') {
                node.status = 'active';
            }
        });
        
        // Randomly assign adversarial status
        const indices = Array.from({length: this.state.nodeStatus.length}, (_, i) => i)
            .sort(() => Math.random() - 0.5)
            .slice(0, adversarialCount);
            
        indices.forEach(index => {
            this.state.nodeStatus[index].status = 'adversarial';
            this.state.nodeStatus[index].lastUpdate = Date.now();
        });
        
        this.updateNodeCounts();
        this.renderNodeGrid();
    }

    updateNodeCounts() {
        const counts = { active: 0, inactive: 0, adversarial: 0 };
        this.state.nodeStatus.forEach(node => {
            counts[node.status]++;
        });
        
        document.getElementById('active-count').textContent = counts.active;
        document.getElementById('inactive-count').textContent = counts.inactive;
        document.getElementById('adversarial-count').textContent = counts.adversarial;
    }

    renderNodeGrid() {
        const grid = document.getElementById('node-grid');
        const filter = document.getElementById('node-filter').value;
        
        grid.innerHTML = '';
        
        this.state.nodeStatus.forEach(node => {
            if (filter !== 'all' && node.status !== filter) return;
            
            const nodeElement = document.createElement('div');
            nodeElement.className = `node-item ${node.status}`;
            nodeElement.title = `Node ${node.id}: ${node.status}`;
            nodeElement.addEventListener('click', () => {
                console.log(`Node ${node.id}:`, node);
            });
            
            grid.appendChild(nodeElement);
        });
        
        this.updateNodeCounts();
    }

    filterNodes(filter) {
        this.renderNodeGrid();
    }

    updateUpdateRate() {
        const now = Date.now();
        const timeDiff = now - this.lastUpdate;
        if (timeDiff > 0) {
            this.updateRate = Math.round(1000 / timeDiff * 10) / 10;
            document.getElementById('update-rate').textContent = this.updateRate;
        }
        this.lastUpdate = now;
    }

    startHeartbeat() {
        setInterval(() => {
            if (this.websocket.connected) {
                const memoryUsage = (45.2 + Math.random() * 10).toFixed(1);
                document.getElementById('memory-usage').textContent = memoryUsage + ' MB';
            }
        }, 2000);
    }

    startProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        this.progressInterval = setInterval(() => {
            if (!this.isRunning) return;
            
            const progress = (this.currentTime / this.totalDuration) * 100;
            document.getElementById('progress-fill').style.width = Math.min(progress, 100) + '%';
        }, 100);
    }

    updateDashboard() {
        // This will be called by updateDashboardAnimated during simulation
        if (!this.isRunning) {
            const metrics = this.state.metrics;
            document.getElementById('performance-score').textContent = metrics.performanceScore.toFixed(1);
            document.getElementById('energy-efficiency').textContent = metrics.energyEfficiency.toFixed(1) + ' pJ';
            document.getElementById('spectrum-utilization').textContent = (metrics.spectrumUtilization * 100).toFixed(1) + '%';
            document.getElementById('active-nodes').textContent = metrics.activeNodes.toString();
        }
    }

    updateCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.update();
            }
        });
    }

    handleWebSocketMessage(message) {
        // Simulate WebSocket message handling
        console.log('📨 WebSocket Message:', message.type, message);
        
        switch (message.type) {
            case 'parameter_changed':
                console.log(`Parameter ${message.parameter} updated to ${message.value}`);
                break;
            case 'simulation_data':
                // Handle real-time data
                break;
            case 'export_request':
                console.log('Export requested:', message.format);
                break;
        }
    }

    // Export functionality
    exportData(format) {
        console.log(`📤 Exporting data in ${format} format...`);
        
        switch (format) {
            case 'json':
                this.exportJSON();
                break;
            case 'csv':
                this.exportCSV();
                break;
            case 'nodes':
                this.exportNodes();
                break;
            case 'config':
                this.exportConfig();
                break;
        }
    }

    exportJSON() {
        const exportData = {
            metadata: {
                exportTime: new Date().toISOString(),
                simulationTime: this.currentTime,
                totalDuration: this.totalDuration,
                isRunning: this.isRunning,
                version: '1.0.0'
            },
            configuration: this.state.config,
            currentMetrics: this.state.metrics,
            timelineData: this.state.timeline,
            nodeStatus: this.state.nodeStatus.slice(0, 100) // First 100 nodes for file size
        };
        
        this.downloadFile('simulation_data.json', JSON.stringify(exportData, null, 2));
        console.log('✅ JSON export completed');
    }

    exportCSV() {
        if (this.state.timeline.length === 0) {
            alert('No timeline data available. Start simulation to generate data.');
            return;
        }
        
        let csv = 'Timestamp,Performance_Score,Learning_Accuracy,Spectrum_Utilization,Energy_Efficiency,Adversarial_Impact,Throughput_Mbps,Active_Nodes,Spike_Rate,Energy_Harvested_uJ\n';
        
        this.state.timeline.forEach(point => {
            csv += [
                point.timestamp,
                point.performanceScore.toFixed(2),
                point.learningAccuracy.toFixed(4),
                point.spectrumUtilization.toFixed(4),
                point.energyEfficiency.toFixed(2),
                point.adversarialImpact.toFixed(4),
                (point.throughput / 1e6).toFixed(2),
                Math.round(point.activeNodes),
                point.spikeRate.toFixed(2),
                (point.energyHarvested * 1e6).toFixed(4)
            ].join(',') + '\n';
        });
        
        this.downloadFile('simulation_timeline.csv', csv);
        console.log('✅ CSV export completed');
    }

    exportNodes() {
        let csv = 'Node_ID,Status,Last_Update\n';
        
        this.state.nodeStatus.forEach(node => {
            csv += `${node.id},${node.status},${new Date(node.lastUpdate).toISOString()}\n`;
        });
        
        this.downloadFile('node_status.csv', csv);
        console.log('✅ Node data export completed');
    }

    exportConfig() {
        const configData = {
            configuration: this.state.config,
            exportTime: new Date().toISOString(),
            simulationState: {
                currentTime: this.currentTime,
                totalDuration: this.totalDuration,
                isRunning: this.isRunning
            }
        };
        
        this.downloadFile('simulation_config.json', JSON.stringify(configData, null, 2));
        console.log('✅ Configuration export completed');
    }

    downloadFile(filename, content) {
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        window.URL.revokeObjectURL(url);
        console.log(`📁 File downloaded: ${filename}`);
    }

    adjustLearningRate(newRate) {
        // Adjust the learning convergence based on new learning rate
        console.log(`📈 Adjusting learning rate to ${newRate}`);
    }

    updateEnergyMetrics() {
        // Recalculate energy metrics based on selected sources
        const sourcesCount = this.state.config.energySources.length;
        this.state.metrics.energyEfficiency = 23.6 * (1 + sourcesCount * 0.1);
        console.log(`⚡ Energy metrics updated for ${sourcesCount} sources`);
    }
}

// Initialize the Real-Time Simulator
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌟 Starting Real-Time Neuromorphic Spectrum Intelligence Simulator...');
    
    window.simulator = new RealTimeNeuromorphicSimulator();
    
    console.log('🎉 Real-Time Simulator Ready!');
    console.log('🔗 WebSocket simulation active');
    console.log('📊 Charts initialized with live streaming');
    console.log('💾 Export functionality enabled');
    console.log('⚙️ Real-time parameter control active');
});