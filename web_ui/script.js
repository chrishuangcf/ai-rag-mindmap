document.addEventListener('DOMContentLoaded', () => {
    // Collapsible Section Functionality
    window.toggleSection = function(contentId) {
        const content = document.getElementById(contentId);
        const button = event.target;
        const container = content.closest('.container'); // Find the parent container

        if (content.classList.contains('collapsed')) {
            content.classList.remove('collapsed');
            if (container) container.classList.remove('collapsed');
            button.textContent = '▼';
            button.classList.remove('collapsed');
        } else {
            content.classList.add('collapsed');
            if (container) container.classList.add('collapsed');
            button.textContent = '▶';
            button.classList.add('collapsed');
        }
    };

    // Initialize all sections as collapsed except RAG a Document and Cache Management
    const collapsibleSections = [
        { id: 'api-health-content', collapsed: true },
        { id: 'rag-document-content', collapsed: false },
        { id: 'cache-management-content', collapsed: false },
        { id: 'mindmap-content', collapsed: true },
        { id: 'url-analysis-content', collapsed: true },
        { id: 'llm-switcher-content', collapsed: true }
    ];
    
    // Set initial state
    collapsibleSections.forEach(section => {
        const content = document.getElementById(section.id);
        const button = document.querySelector(`[onclick="toggleSection('${section.id}')"]`);
        const container = content.closest('.container');
        
        if (content && button) {
            if (section.collapsed) {
                content.classList.add('collapsed');
                if (container) container.classList.add('collapsed');
                button.textContent = '▶';
                button.classList.add('collapsed');
            } else {
                content.classList.remove('collapsed');
                if (container) container.classList.remove('collapsed');
                button.textContent = '▼';
                button.classList.remove('collapsed');
            }
        }
    });

    // LLM Provider Elements
    const providerSelect = document.getElementById('provider-select');
    const currentProvider = document.getElementById('current-provider');
    const currentModel = document.getElementById('current-model');

    // Cache Management Elements
    const cacheTableBody = document.querySelector('#cacheTable tbody');
    const searchInput = document.getElementById('searchInput');
    const clearAllButton = document.getElementById('clearAllButton');

    // Health Check Elements
    const healthCheckButton = document.getElementById('healthCheckButton');
    const healthStatus = document.getElementById('healthStatus');
    const systemStatus = document.getElementById('systemStatus');
    const autoRefreshToggle = document.getElementById('autoRefreshToggle');

    // RAG Form Elements
    const ragForm = document.getElementById('ragForm');
    const repoUrlsInput = document.getElementById('repoUrls');
    const uploadForm = document.getElementById('uploadForm');
    const fileUploadInput = document.getElementById('fileUpload');
    const ragResponseContainer = document.getElementById('ragResponseContainer');
    const ragAnswer = document.getElementById('ragAnswer');
    const ragSources = document.getElementById('ragSources');

    // Global Search Elements
    const globalSearchForm = document.getElementById('globalSearchForm');
    const globalQueryInput = document.getElementById('globalQuery');
    const globalSearchResults = document.getElementById('globalSearchResults');
    const searchAnswer = document.getElementById('searchAnswer');
    const searchResults = document.getElementById('searchResults');
    const searchResultsTitle = document.getElementById('searchResultsTitle');
    const globalSourcesDetails = document.getElementById('globalSourcesDetails');
    const globalSourcesToggle = document.getElementById('globalSourcesToggle');

    const API_BASE_URL = '/api';

    // --- Health Check ---
    const checkHealth = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            let mainServiceData = null;
            
            if (response.ok) {
                mainServiceData = await response.json();
                healthStatus.innerHTML = `<span class="healthy">Main Service: ${mainServiceData.status}</span>`;
            } else {
                healthStatus.innerHTML = '<span class="unhealthy">Main Service: Unhealthy</span>';
            }

            // Check mindmap service health
            let mindmapServiceData = null;
            try {
                console.log('Checking mindmap service health...');
                const mindmapResponse = await fetch('http://localhost:8003/health');
                if (mindmapResponse.ok) {
                    mindmapServiceData = await mindmapResponse.json();
                    console.log('Mindmap service data:', mindmapServiceData);
                } else {
                    console.log('Mindmap service responded with status:', mindmapResponse.status);
                }
            } catch (mindmapError) {
                console.error('Error checking mindmap service:', mindmapError);
            }

            // Also fetch cache status to show in-memory cache info
            let cacheData = null;
            try {
                const cacheResponse = await fetch(`${API_BASE_URL}/debug/cache-status`);
                if (cacheResponse.ok) {
                    cacheData = await cacheResponse.json();
                }
            } catch (cacheError) {
                console.error('Error fetching cache status:', cacheError);
            }

            // Show comprehensive system status
            const systemStatusEl = document.getElementById('systemStatus');
            let systemStatusHTML = '<div class="system-info">';
            
            // Main service components (excluding Google Search for mindmap service status)
            if (mainServiceData) {
                systemStatusHTML += `
                    <span class="${mainServiceData.redis_available ? 'healthy' : 'unhealthy'}">
                        Redis: ${mainServiceData.redis_available ? 'Available' : 'Unavailable'}
                    </span>
                    <span class="${mainServiceData.redis_search_available ? 'healthy' : 'unhealthy'}">
                        Redis Search: ${mainServiceData.redis_search_available ? 'Available' : 'Unavailable'}
                    </span>
                `;
                
                if (cacheData) {
                    systemStatusHTML += `
                        <span class="${cacheData.total_memory_caches > 0 ? 'healthy' : 'unhealthy'}">
                            In-Memory Cache: ${cacheData.total_memory_caches} items
                        </span>
                    `;
                } else {
                    systemStatusHTML += `
                        <span class="unhealthy">
                            In-Memory Cache: Error fetching status
                        </span>
                    `;
                }
            } else {
                systemStatusHTML += '<span class="unhealthy">Main service status unavailable</span>';
            }

            systemStatusHTML += '</div>';

            // Mindmap service status section
            systemStatusHTML += '<div class="mindmap-service-info">';
            systemStatusHTML += '<h4>Mind Map Service Status</h4>';
            
            if (mindmapServiceData) {
                const overallStatus = mindmapServiceData.service === 'healthy' ? 'healthy' : 
                                     mindmapServiceData.service === 'degraded' ? 'processing' : 'unhealthy';
                systemStatusHTML += `
                    <div class="service-status">
                        <span class="${overallStatus}">
                            Service: ${mindmapServiceData.service}
                        </span>
                        <span class="${mindmapServiceData.neo4j === 'healthy' ? 'healthy' : 'unhealthy'}">
                            Neo4j: ${mindmapServiceData.neo4j}
                        </span>
                        <span class="${mindmapServiceData.redis === 'healthy' ? 'healthy' : 'unhealthy'}">
                            Redis: ${mindmapServiceData.redis}
                        </span>
                `;

                // Processing status with detailed information
                const processingStatus = mindmapServiceData.processing_status;
                let processingClass = '';
                let processingText = '';
                let processingIcon = '';

                switch (processingStatus) {
                    case 'processing':
                        processingClass = 'processing';
                        processingText = 'Processing Tasks';
                        processingIcon = '⚙️';
                        break;
                    case 'pending':
                        processingClass = 'processing';
                        processingText = 'Tasks Pending';
                        processingIcon = '⏳';
                        break;
                    case 'idle':
                        processingClass = 'healthy';
                        processingText = 'Idle';
                        processingIcon = '💤';
                        break;
                    case 'stopped':
                        processingClass = 'unhealthy';
                        processingText = 'Stopped';
                        processingIcon = '⏹️';
                        break;
                    case 'error':
                        processingClass = 'unhealthy';
                        processingText = 'Error';
                        processingIcon = '❌';
                        break;
                    default:
                        processingClass = 'unhealthy';
                        processingText = 'Unknown';
                        processingIcon = '❓';
                }

                systemStatusHTML += `
                        <span class="${processingClass}">
                            ${processingIcon} Processing: ${processingText}
                        </span>
                    </div>
                `;

                // Show detailed queue information
                if (mindmapServiceData.queue_info) {
                    const queueInfo = mindmapServiceData.queue_info;
                    systemStatusHTML += `
                        <div class="queue-details">
                            <div class="queue-stats">
                                <span class="queue-stat">
                                    📊 Running: ${queueInfo.running_jobs}/${queueInfo.max_concurrent}
                                </span>
                                <span class="queue-stat">
                                    📝 Pending: ${queueInfo.pending_jobs}
                                </span>
                                <span class="queue-stat">
                                    ✅ Completed: ${queueInfo.completed_jobs}
                                </span>
                                <span class="queue-stat ${queueInfo.failed_jobs > 0 ? 'unhealthy' : 'healthy'}">
                                    ❌ Failed: ${queueInfo.failed_jobs}
                                </span>
                                <span class="queue-stat">
                                    📈 Success: ${queueInfo.success_rate}%
                                </span>
                            </div>
                        </div>
                    `;
                }

                // Show processing details if actively processing
                if (processingStatus === 'processing' || processingStatus === 'pending') {
                    systemStatusHTML += `
                        <div class="processing-details">
                            <small>🔄 Mind map service is ${processingStatus === 'processing' ? 'actively processing' : 'preparing to process'} tasks...</small>
                        </div>
                    `;
                }
            } else {
                systemStatusHTML += `
                    <div class="service-status">
                        <span class="unhealthy">
                            ❌ Mind Map Service: Unavailable
                        </span>
                        <span class="unhealthy">
                            Cannot connect to mind map service (localhost:8003)
                        </span>
                    </div>
                `;
            }
            
            systemStatusHTML += '</div>';
            systemStatusEl.innerHTML = systemStatusHTML;

            // Update API Health status dot
            updateApiHealthStatusDot(mainServiceData, mindmapServiceData);

            // Update mindmap status based on service processing status
            updateMindmapProcessingStatus(mindmapServiceData);

        } catch (error) {
            console.error('Error checking health:', error);
            healthStatus.innerHTML = '<span class="unhealthy">Status: Error</span>';
            const systemStatusEl = document.getElementById('systemStatus');
            systemStatusEl.innerHTML = '<div class="system-info"><span class="unhealthy">Connection error</span></div>';
        }
    };

    // Update API Health status dot function
    const updateApiHealthStatusDot = (mainServiceData, mindmapServiceData) => {
        const statusDot = document.getElementById('api-health-status-dot');
        if (!statusDot) {
            console.error('Status dot element not found');
            return;
        }

        // Check if both Redis Search and Mindmap Service are available
        const redisSearchAvailable = mainServiceData && mainServiceData.redis_search_available;
        const mindmapServiceAvailable = mindmapServiceData && (mindmapServiceData.service === 'healthy' || mindmapServiceData.service === 'degraded');

        console.log('Status check:', { redisSearchAvailable, mindmapServiceAvailable, mainServiceData, mindmapServiceData });

        if (redisSearchAvailable && mindmapServiceAvailable) {
            statusDot.textContent = '🟢';
            statusDot.className = 'status-dot healthy';
            statusDot.style.display = 'inline';
            statusDot.title = 'Redis Search and Mind Map Service are both healthy';
        } else {
            statusDot.textContent = '🔴';
            statusDot.className = 'status-dot unhealthy';
            statusDot.style.display = 'inline';
            
            let issues = [];
            if (!redisSearchAvailable) issues.push('Redis Search unavailable');
            if (!mindmapServiceAvailable) issues.push('Mind Map Service unhealthy');
            statusDot.title = issues.join(', ');
        }
    };

    // Update mindmap status based on service processing status
    const updateMindmapProcessingStatus = (mindmapServiceData) => {
        // These elements might not exist in the current UI
        const mindmapStatusEl = document.getElementById('mindmapStatus');
        const refreshButton = document.getElementById('refreshMindmapButton');

        // Safely handle missing elements
        if (!mindmapStatusEl && !refreshButton) return;

        // Protect against null properties
        const safeSetInnerHTML = (el, content) => el && (el.innerHTML = content);
        const safeSetDisabled = (el, state) => el && (el.disabled = state);
        const safeSetTitle = (el, text) => el && (el.title = text);
        
        if (!mindmapServiceData) {
            safeSetInnerHTML(mindmapStatusEl, `
                <span class="mindmap-status-offline">
                    🔌 Mind Map Service Offline - Cannot refresh visualization
                </span>
            `);
            safeSetDisabled(refreshButton, true);
            safeSetTitle(refreshButton, "Mind Map Service is offline");
            return;
        }

        const processingStatus = mindmapServiceData.processing_status;
        
        if (processingStatus === 'processing') {
            safeSetInnerHTML(mindmapStatusEl, `
                <span class="mindmap-status-processing">
                    ⚙️ Mind maps are being processed in background...
                </span>
            `);
            if (refreshButton) {
                refreshButton.innerHTML = '🔄 Processing...';
                safeSetDisabled(refreshButton, true);
            }
        } else if (processingStatus === 'pending') {
            safeSetInnerHTML(mindmapStatusEl, `
                <span class="mindmap-status-pending">
                    ⏳ Mind map tasks are queued for processing...
                </span>
            `);
            if (refreshButton) {
                refreshButton.innerHTML = '⏳ Tasks Queued';
                safeSetDisabled(refreshButton, true);
            }
        } else {
            // Service is idle/healthy - restore normal functionality
            if (refreshButton) {
                refreshButton.innerHTML = '🔄 Refresh Mind Map';
                refreshButton.disabled = false;
                refreshButton.title = "";
            }
        }
    };

    // Auto-refresh health status every 10 seconds
    let healthCheckInterval = null;
    const startHealthMonitoring = () => {
        // Initial check
        checkHealth();
        
        // Set up auto-refresh
        if (healthCheckInterval) {
            clearInterval(healthCheckInterval);
        }
        
        healthCheckInterval = setInterval(checkHealth, 10000); // Every 10 seconds
    };

    const stopHealthMonitoring = () => {
        if (healthCheckInterval) {
            clearInterval(healthCheckInterval);
            healthCheckInterval = null;
        }
    };

    // --- RAG Functionality ---
    const handleRagSubmit = async (event) => {
        event.preventDefault();
        const urls = repoUrlsInput.value.split('\n').map(url => url.trim()).filter(url => url);
        const gitlabToken = document.getElementById('gitlabToken').value;

        if (urls.length === 0) {
            alert('Please provide at least one URL.');
            return;
        }

        ragResponseContainer.style.display = 'block';
        ragAnswer.textContent = 'Processing document(s)...';
        ragSources.textContent = '';

        try {
            const requestBody = { repo_urls: urls };
            
            // Add GitLab token if provided
            if (gitlabToken.trim()) {
                requestBody.gitlab_token = gitlabToken.trim();
            }

            const response = await fetch(`${API_BASE_URL}/rag`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An unknown error occurred.');
            }

            ragAnswer.innerHTML = `<p><strong>Document(s) processed successfully!</strong></p>
                                    <p>Cache hash: <code>${data.cache_hash}</code></p>
                                    <p>Total chunks created: ${data.total_chunks}</p>`;
            
            ragSources.innerHTML = ''; // Clear previous sources
            data.processed_urls.forEach(url => {
                const sourceEl = document.createElement('div');
                sourceEl.className = 'source-document';
                sourceEl.innerHTML = `<a href="${url}" target="_blank">${url}</a>`;
                ragSources.appendChild(sourceEl);
            });
            
            // Update source toggle with count
            const ragSourcesToggle = document.getElementById('ragSourcesToggle');
            ragSourcesToggle.textContent = `View Source Documents (${data.processed_urls.length})`;
            
            // Refresh cache list as it might have been updated
            fetchAndRenderCaches();
            
            // Refresh mind map visualizer caches and auto-generate if enabled
            if (window.vectorMindMapVisualizer) {
                if (document.getElementById('auto-mindmap-enabled')?.checked) {
                    setTimeout(() => {
                        window.vectorMindMapVisualizer.tryAutoGeneration();
                    }, 1000); // Give time for cache to be saved
                } else {
                    // Even if auto-generation is disabled, refresh the available caches
                    setTimeout(() => {
                        window.vectorMindMapVisualizer.refreshCaches();
                    }, 1000);
                }
            }

        } catch (error) {
            console.error('Error with RAG request:', error);
            ragAnswer.innerHTML = `<p>Error: ${error.message}</p>`;
        }
    };

    const handleFileUpload = async (event) => {
        event.preventDefault();
        const files = fileUploadInput.files;

        if (!files || files.length === 0) {
            alert('Please select at least one file to upload.');
            return;
        }

        ragResponseContainer.style.display = 'block';
        const fileNames = Array.from(files).map(f => f.name).join(', ');
        ragAnswer.innerHTML = `<p>Uploading and processing ${files.length} file(s): <strong>${fileNames}</strong>...</p>`;
        ragSources.textContent = '';

        const formData = new FormData();
        // Append all files to FormData
        for (const file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch(`${API_BASE_URL}/rag/upload`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An unknown error occurred during file upload.');
            }

            // Display results for multiple files
            let resultHtml = `<p><strong>Files processed successfully!</strong></p>
                             <p>Cache hash: <code>${data.cache_hash}</code></p>
                             <p>Total chunks created: ${data.total_chunks}</p>
                             <p>Processed files: ${data.processed_files.length}</p>`;
            
            if (data.skipped_files && data.skipped_files.length > 0) {
                resultHtml += `<p class="warning">Skipped files: ${data.skipped_files.join(', ')}</p>`;
            }
            
            ragAnswer.innerHTML = resultHtml;
            
            ragSources.innerHTML = ''; // Clear previous sources
            data.processed_files.forEach(filename => {
                const sourceEl = document.createElement('div');
                sourceEl.className = 'source-document';
                sourceEl.textContent = filename;
                ragSources.appendChild(sourceEl);
            });
            
            const ragSourcesToggle = document.getElementById('ragSourcesToggle');
            ragSourcesToggle.textContent = `View Source Documents (${data.processed_files.length})`;

            fetchAndRenderCaches();
            
            // Refresh mind map visualizer caches and auto-generate if enabled
            if (window.vectorMindMapVisualizer) {
                if (document.getElementById('auto-mindmap-enabled')?.checked) {
                    setTimeout(() => {
                        window.vectorMindMapVisualizer.tryAutoGeneration();
                    }, 1000); // Give time for cache to be saved
                } else {
                    // Even if auto-generation is disabled, refresh the available caches
                    setTimeout(() => {
                        window.vectorMindMapVisualizer.refreshCaches();
                    }, 1000);
                }
            }

        } catch (error) {
            console.error('Error with file upload:', error);
            ragAnswer.innerHTML = `<p>Error: ${error.message}</p>`;
        }
    };

    // --- Test GitLab Token ---
    const handleTestGitlab = async () => {
        const gitlabToken = document.getElementById('gitlabToken').value;
        const testButton = document.querySelector('button[onclick="handleTestGitlab()"]');
        
        if (!gitlabToken.trim()) {
            alert('Please enter a GitLab access token first');
            return;
        }

        // Show loading state
        const originalText = testButton.textContent;
        testButton.textContent = 'Testing...';
        testButton.disabled = true;

        try {
            // Test the token by making a simple API call to GitLab
            const testUrl = 'https://gitlab.com/api/v4/user';
            const response = await fetch(`${API_BASE_URL}/test-gitlab-token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    token: gitlabToken.trim(),
                    test_url: testUrl
                })
            });

            const data = await response.json();

            if (response.ok) {
                alert(`✅ GitLab token is valid!\n\nUser: ${data.user_info?.name || 'Unknown'}\nUsername: ${data.user_info?.username || 'Unknown'}`);
            } else {
                alert(`❌ GitLab token test failed:\n${data.detail || 'Unknown error'}`);
            }

        } catch (error) {
            console.error('Error testing GitLab token:', error);
            alert(`❌ Error testing GitLab token:\n${error.message}`);
        } finally {
            // Restore button state
            testButton.textContent = originalText;
            testButton.disabled = false;
        }
    };

    // Make handleTestGitlab globally available
    window.handleTestGitlab = handleTestGitlab;

    // --- Cache Management ---
    const fetchAndRenderCaches = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/cache/list`);
            const data = await response.json();
            renderCacheTable(data.available_caches);
        } catch (error) {
            console.error('Error fetching caches:', error);
            cacheTableBody.innerHTML = '<tr><td colspan="7">Error loading caches. Is the backend running?</td></tr>';
        }
    };

    const renderCacheTable = (caches) => {
        cacheTableBody.innerHTML = '';
        if (!caches || caches.length === 0) {
            cacheTableBody.innerHTML = '<tr><td colspan="7">No caches found.</td></tr>';
            return;
        }

        const filteredCaches = (caches || []).filter(cache => {
            const searchTerm = searchInput.value.toLowerCase();
            if (!searchTerm) return true;
            const urls = cache.urls.join(' ').toLowerCase();
            const urlPreview = cache.url_preview.join(' ').toLowerCase();
            return urls.includes(searchTerm) || urlPreview.includes(searchTerm);
        });

        if (filteredCaches.length === 0) {
            cacheTableBody.innerHTML = '<tr><td colspan="7">No caches match your search.</td></tr>';
            return;
        }

        filteredCaches.forEach(cache => {
            const row = document.createElement('tr');
            
            // Format storage locations
            const storageDisplay = cache.storage_locations ? cache.storage_locations.join(', ') : 'Unknown';
            
            // Make URLs clickable for cache content viewing
            const clickableUrls = cache.urls.map(url => 
                `<span class="cache-url" data-hash="${cache.hash}" data-url="${url}" style="color: #cc0066; cursor: pointer; text-decoration: underline;">${url}</span>`
            ).join('<br>');
            
            row.innerHTML = `
                <td>${cache.hash}</td>
                <td>${clickableUrls}</td>
                <td>${storageDisplay}</td>
                <td>${cache.total_chunks}</td>
                <td>${new Date(cache.created_at * 1000).toLocaleString()}</td>
                <td>${new Date(cache.last_accessed * 1000).toLocaleString()}</td>
                <td><button class="delete-button" data-hash="${cache.hash}">Delete</button></td>
            `;
            cacheTableBody.appendChild(row);
        });
    };

    const handleDeleteCache = async (event) => {
        if (event.target.classList.contains('delete-button')) {
            const hash = event.target.dataset.hash;
            if (confirm(`Are you sure you want to delete cache ${hash}?`)) {
                try {
                    await fetch(`${API_BASE_URL}/cache/${hash}`, { method: 'DELETE' });
                    fetchAndRenderCaches();
                    // Also refresh the mind map's internal cache list
                    if (window.vectorMindMapVisualizer) {
                        window.vectorMindMapVisualizer.refreshCaches();
                    }
                } catch (error) {
                    console.error('Error deleting cache:', error);
                    alert('Failed to delete cache.');
                }
            }
        }
    };

    const handleCacheClick = async (event) => {
        if (event.target.classList.contains('cache-url')) {
            const hash = event.target.dataset.hash;
            const url = event.target.dataset.url;
            await showCacheContent(hash, url);
        }
    };

    const handleClearAll = async () => {
        if (confirm('Are you sure you want to delete all caches? This action cannot be undone.')) {
            try {
                await fetch(`${API_BASE_URL}/cache`, { method: 'DELETE' });
                fetchAndRenderCaches();
            } catch (error) {
                console.error('Error clearing all caches:', error);
                alert('Failed to clear all caches.');
            }
        }
    };

    // --- Global Search Functionality ---
    const handleGlobalSearch = async (event) => {
        event.preventDefault();
        const query = globalQueryInput.value.trim();
        const analysisTypeRadio = document.querySelector('input[name="analysisType"]:checked');
        const detailedAnalysis = analysisTypeRadio ? analysisTypeRadio.value === 'deep' : false;

        if (!query) {
            alert('Please enter a search query.');
            return;
        }

        globalSearchResults.style.display = 'block';
        
        // Show analysis type and start timing
        const startTime = Date.now();
        const analysisType = detailedAnalysis ? 'Deep' : 'Summary';
        searchResultsTitle.innerHTML = `AI Answer <span class="analysis-type-badge analysis-type-${detailedAnalysis ? 'deep' : 'summary'}">${analysisType} Analysis</span> <span class="processing-time">Processing...</span>`;
        searchAnswer.textContent = 'Processing...';
        searchAnswer.style.display = 'block';
        searchResults.innerHTML = '';
        
        // Make sure the details element is closed by default for AI answers
        const globalSourcesDetails = document.getElementById('globalSourcesDetails');
        globalSourcesDetails.open = false;

        try {
            // Use the global endpoint with different parameters based on analysis type
            const response = await fetch(`${API_BASE_URL}/rag/global`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query,
                    max_results: detailedAnalysis ? 30 : 20,
                    max_docs_per_cache: detailedAnalysis ? 8 : 5,
                    detailed_analysis: detailedAnalysis  // Flag for backend to use enhanced prompts
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Search failed');
            }

            // Calculate processing time and update title
            const endTime = Date.now();
            const processingTime = ((endTime - startTime) / 1000).toFixed(1);
            searchResultsTitle.innerHTML = `AI Answer <span class="analysis-type-badge analysis-type-${detailedAnalysis ? 'deep' : 'summary'}">${analysisType} Analysis</span> <span class="processing-time">(${processingTime}s)</span>`;

            // Display AI answer
            searchAnswer.style.display = 'block';
            searchAnswer.innerHTML = marked.parse(data.answer);
            
            // Apply Google fallback theme if Google search was used
            if (data.google_fallback) {
                globalSearchResults.classList.add('google-enhanced');
            } else {
                globalSearchResults.classList.remove('google-enhanced');
            }
            
            // Update source toggle with count and keep it collapsed
            const globalSourcesToggle = document.getElementById('globalSourcesToggle');
            globalSourcesToggle.textContent = `View Source Documents (${data.source_documents.length})`;
            
            // Display source information in collapsible section
            let metaInfo = `
                <div class="search-meta">
                    <p><strong>Searched ${data.searched_caches.length} document sets</strong></p>
                    <p><strong>Found ${data.total_documents_found} relevant documents</strong></p>
            `;
            
            // Add Google fallback indicator if present
            if (data.google_fallback) {
                metaInfo += `<p class="google-fallback"><strong>🔍 Enhanced with Google Search results (AI didn't know from documents)</strong></p>`;
            } else if (data.google_fallback_attempted) {
                metaInfo += `<p class="google-fallback-failed"><strong>🔍 Google Search was attempted but returned no results</strong></p>`;
            }
            
            metaInfo += `</div>`;
            searchResults.innerHTML = metaInfo;
            
            // Display sources
            data.source_documents.forEach(doc => {
                const sourceEl = document.createElement('div');
                sourceEl.className = 'source-document';
                
                // Add special styling for Google results
                if (doc.storage_type === 'google') {
                    sourceEl.classList.add('google-result');
                    sourceEl.innerHTML = `
                        <h4>🔍 Google Result: <a href="${doc.source_url}" target="_blank">${doc.source_url}</a></h4>
                        <p><em>Source: Google Search</em></p>
                        <pre>${doc.content}</pre>
                    `;
                } else {
                    sourceEl.innerHTML = `
                        <h4>Source: <a href="${doc.source_url}" target="_blank">${doc.source_url}</a></h4>
                        <p><em>From cache: ${doc.cache_hash} (${doc.cache_urls.length} files)</em></p>
                        <pre>${doc.content}</pre>
                    `;
                }
                searchResults.appendChild(sourceEl);
            });

        } catch (error) {
            console.error('Error with global search:', error);
            
            // Calculate processing time even for errors and update title
            const endTime = Date.now();
            const processingTime = ((endTime - startTime) / 1000).toFixed(1);
            searchResultsTitle.innerHTML = `AI Answer <span class="analysis-type-badge analysis-type-${detailedAnalysis ? 'deep' : 'summary'}">${analysisType} Analysis</span> <span class="processing-time error">(${processingTime}s - Error)</span>`;
            
            searchResults.innerHTML = `<p>Error: ${error.message}</p>`;
            // Reset Google theme on error
            globalSearchResults.classList.remove('google-enhanced');
        }
    };

    // --- URL Analysis Handler ---
    const handleUrlAnalysis = async (event) => {
        event.preventDefault();
        const urlsText = document.getElementById('urlAnalysisUrls').value.trim();
        const query = document.getElementById('urlAnalysisQuery').value.trim();
        const analysisTypeRadio = document.querySelector('input[name="urlAnalysisType"]:checked');
        const detailed = analysisTypeRadio ? analysisTypeRadio.value === 'deep' : false;

        if (!urlsText || !query) {
            alert('Please enter both URLs and a question.');
            return;
        }

        const urls = urlsText.split('\n').map(url => url.trim()).filter(url => url);
        if (urls.length === 0) {
            alert('Please enter at least one valid URL.');
            return;
        }

        const urlAnalysisResults = document.getElementById('urlAnalysisResults');
        const urlAnalysisAnswer = document.getElementById('urlAnalysisAnswer');
        const urlAnalysisSources = document.getElementById('urlAnalysisSources');

        urlAnalysisResults.style.display = 'block';
        const startTime = Date.now();
        const analysisType = detailed ? 'Deep' : 'Summary';
        urlAnalysisAnswer.innerHTML = `<p>Processing ${urls.length} URL(s) with ${analysisType.toLowerCase()} analysis...</p>`;
        urlAnalysisSources.textContent = '';

        try {
            const response = await fetch(`${API_BASE_URL}/rag/url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    urls: urls, 
                    query: query,
                    detailed: detailed 
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An unknown error occurred.');
            }

            // Calculate processing time
            const endTime = Date.now();
            const processingTime = ((endTime - startTime) / 1000).toFixed(1);

            // Add analysis type badge with processing time
            let answerHtml = `<div class="analysis-type-badge analysis-type-${data.analysis_type}">${data.analysis_type.charAt(0).toUpperCase() + data.analysis_type.slice(1)} Analysis <span class="processing-time">(${processingTime}s)</span></div>`;
            
            // Handle Google fallback response
            if (data.google_fallback) {
                answerHtml += `
                    <div class="google-fallback-notice">
                        <p><strong>🔍 Answer from Google Search (URL content didn't contain relevant information):</strong></p>
                    </div>
                    ${marked.parse(data.answer)}
                    <div class="original-rag-notice">
                        <details>
                            <summary>View original URL analysis</summary>
                            <div>${marked.parse(data.original_rag_answer)}</div>
                        </details>
                    </div>
                `;
            } else {
                answerHtml += marked.parse(data.answer);
            }

            urlAnalysisAnswer.innerHTML = answerHtml;

            // Display source documents
            urlAnalysisSources.innerHTML = '';
            if (data.source_documents && data.source_documents.length > 0) {
                data.source_documents.forEach(doc => {
                    const sourceEl = document.createElement('div');
                    sourceEl.className = 'source-document';
                    if (doc.storage_type === 'google') {
                        sourceEl.innerHTML = `<strong>Google Result:</strong> <a href="${doc.source_url}" target="_blank">${doc.source_url}</a><br><em>${doc.content.substring(0, 300)}...</em>`;
                    } else {
                        sourceEl.innerHTML = `<strong>Source:</strong> <a href="${doc.source_url}" target="_blank">${doc.source_url}</a><br><em>${doc.content.substring(0, 300)}...</em>`;
                    }
                    urlAnalysisSources.appendChild(sourceEl);
                });
                
                const urlAnalysisSourcesToggle = document.getElementById('urlAnalysisSourcesToggle');
                urlAnalysisSourcesToggle.textContent = `View Source Documents (${data.source_documents.length}) - ${data.documents_analyzed || data.source_documents.length} analyzed`;
            }

        } catch (error) {
            console.error('Error with URL analysis:', error);
            
            // Calculate processing time even for errors
            const endTime = Date.now();
            const processingTime = ((endTime - startTime) / 1000).toFixed(1);
            
            urlAnalysisAnswer.innerHTML = `<div class="analysis-type-badge analysis-type-summary">Error <span class="processing-time error">(${processingTime}s)</span></div><p>Error: ${error.message}</p>`;
        }
    };

    // --- Common Modal Utility Functions ---
    
    window.showModalTab = function(tabName) {
        // Update tab buttons
        document.querySelectorAll('.modal-tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`modal-${tabName}-btn`).classList.add('active');
        
        // Update tab content
        document.querySelectorAll('.modal-tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`modal-${tabName}-content`).classList.add('active');
    };

    const showModal = (title, rawContent, markdownContent = null) => {
        const modal = document.getElementById('content-detail-modal');
        const titleEl = document.getElementById('modal-title');
        const rawEl = document.getElementById('modal-raw-content');
        const markdownEl = document.getElementById('modal-markdown-content');
        
        titleEl.textContent = title;
        rawEl.textContent = rawContent;
        
        if (markdownContent) {
            markdownEl.innerHTML = markdownContent;
        } else {
            // Parse markdown using marked.js if available
            if (typeof marked !== 'undefined') {
                markdownEl.innerHTML = marked.parse(rawContent);
            } else {
                markdownEl.innerHTML = `<pre>${rawContent}</pre>`;
            }
        }
        
        // Show raw tab by default
        showModalTab('raw');
        modal.style.display = 'block';
    };

    const showCacheContent = async (hash, url = null) => {
        const modal = document.getElementById('content-detail-modal');
        const titleEl = document.getElementById('modal-title');
        const rawEl = document.getElementById('modal-raw-content');
        const markdownEl = document.getElementById('modal-markdown-content');
        
        titleEl.textContent = url ? `Cache Content: ${url}` : `Cache Content: ${hash}`;
        rawEl.innerHTML = '<p>Loading cache content...</p>';
        markdownEl.innerHTML = '<p>Loading cache content...</p>';
        
        showModalTab('raw');
        modal.style.display = 'block';
        
        try {
            const response = await fetch(`${API_BASE_URL}/cache/${hash}/documents`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.documents || data.documents.length === 0) {
                rawEl.innerHTML = '<p>No documents found in this cache.</p>';
                markdownEl.innerHTML = '<p>No documents found in this cache.</p>';
                return;
            }
            
            // Combine all document content
            let allContent = '';
            let filteredDocuments = data.documents;
            
            // If a specific URL was clicked, filter to show only that URL's content
            if (url) {
                filteredDocuments = data.documents.filter(doc => 
                    doc.metadata.source_url === url
                );
            }
            
            if (filteredDocuments.length === 0) {
                rawEl.innerHTML = `<p>No content found for URL: ${url}</p>`;
                markdownEl.innerHTML = `<p>No content found for URL: ${url}</p>`;
                return;
            }
            
            filteredDocuments.forEach((doc, index) => {
                if (filteredDocuments.length > 1) {
                    allContent += `--- Document ${index + 1}: ${doc.metadata.source_url || 'Unknown'} ---\n\n`;
                }
                allContent += doc.content + '\n\n';
            });
            
            // Show raw content
            rawEl.textContent = allContent;
            
            // Show markdown rendered content
            if (typeof marked !== 'undefined') {
                markdownEl.innerHTML = marked.parse(allContent);
            } else {
                markdownEl.innerHTML = `<pre>${allContent}</pre>`;
            }
            
        } catch (error) {
            console.error('Error fetching cache content:', error);
            const errorMsg = `Error loading cache content: ${error.message}`;
            rawEl.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
            markdownEl.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
        }
    };

    // --- End of Common Modal Functions ---

    // --- Event Listeners ---
    healthCheckButton.addEventListener('click', () => {
        checkHealth();
        // Restart monitoring on manual check
        if (autoRefreshToggle.checked) {
            startHealthMonitoring();
        }
    });

    autoRefreshToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            startHealthMonitoring();
        } else {
            stopHealthMonitoring();
        }
    });
    ragForm.addEventListener('submit', handleRagSubmit);
    uploadForm.addEventListener('submit', handleFileUpload);
    cacheTableBody.addEventListener('click', handleDeleteCache);
    cacheTableBody.addEventListener('click', handleCacheClick); // Add cache URL click handler
    clearAllButton.addEventListener('click', handleClearAll);
    searchInput.addEventListener('input', () => fetchAndRenderCaches());
    globalSearchForm.addEventListener('submit', handleGlobalSearch);
    document.getElementById('urlAnalysisForm').addEventListener('submit', handleUrlAnalysis);

    // --- LLM Provider Switching ---
    const fetchLlmConfig = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/config/llm`);
            if (!response.ok) throw new Error('Failed to fetch LLM config');
            const config = await response.json();
            updateLlmDisplay(config);
        } catch (error) {
            console.error('Error fetching LLM config:', error);
            currentProvider.textContent = 'Error';
            currentModel.textContent = 'Error';
        }
    };

    const updateLlmDisplay = (config) => {
        currentProvider.textContent = config.provider;
        currentModel.textContent = config.model;
        providerSelect.value = config.provider;
    };

    const handleProviderSwitch = async (event) => {
        const selectedProvider = event.target.value;
        try {
            const response = await fetch(`${API_BASE_URL}/config/llm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: selectedProvider }),
            });
            if (!response.ok) throw new Error('Failed to switch provider');
            const newConfig = await response.json();
            updateLlmDisplay(newConfig);
            alert(`Switched to ${newConfig.provider} successfully!`);
        } catch (error) {
            console.error('Error switching LLM provider:', error);
            alert('Failed to switch provider. Check the console for details.');
            // Re-fetch config to revert display to the actual current state
            fetchLlmConfig();
        }
    };

    providerSelect.addEventListener('change', handleProviderSwitch);


    let mindmapSvg, mindmapG, zoom, currentTransform, simulation;

    const initializeMindMap = () => {
        const svg = d3.select('#mindmapSvg');
        const container = document.getElementById('mindmapContainer');
        const width = container.clientWidth;
        const height = 600;

        svg.attr('width', width).attr('height', height);

        // Clear existing content
        svg.selectAll('*').remove();

        // Create zoom behavior
        zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                currentTransform = event.transform;
                mindmapG.attr('transform', currentTransform);
            });

        svg.call(zoom);

        // Create main group for zoomable content
        mindmapG = svg.append('g');
        currentTransform = d3.zoomIdentity;

        // Add zoom controls event listeners
        document.getElementById('zoomInButton').onclick = () => {
            svg.transition().call(zoom.scaleBy, 1.5);
        };

        document.getElementById('zoomOutButton').onclick = () => {
            svg.transition().call(zoom.scaleBy, 1 / 1.5);
        };

        document.getElementById('resetZoomButton').onclick = () => {
            svg.transition().call(zoom.transform, d3.zoomIdentity);
        };

        document.getElementById('refreshMindmapButton').onclick = refreshMindMap;
    };

    const loadMindMapData = async () => {
        try {
            document.getElementById('mindmapStatus').textContent = 'Loading mind map...';
            const response = await fetch(`${API_BASE_URL}/mindmap/unified`);
            const data = await response.json();

            if (data.status === 'success' && data.mindmap) {
                renderMindMap(data.mindmap);
                document.getElementById('mindmapStatus').style.display = 'none';
            } else if (data.status === 'no_data') {
                document.getElementById('mindmapStatus').textContent = 'No documents available for mind map visualization';
            } else {
                document.getElementById('mindmapStatus').textContent = 'Error loading mind map: ' + (data.message || 'Unknown error');
            }
        } catch (error) {
            console.error('Error loading mind map:', error);
            document.getElementById('mindmapStatus').textContent = 'Failed to load mind map';
        }
    };

    const renderMindMap = (mindmapData) => {
        try {
            // Validate mindmap data structure
            if (!mindmapData || !Array.isArray(mindmapData.nodes)) {
                throw new Error('Invalid mindmap data structure');
            }
            
            const nodes = mindmapData.nodes.map(node => ({
                id: node.id || Math.random().toString(36).substr(2, 9),
                name: node.name || 'Unnamed Concept',
                type: node.type || 'general',
                size: node.size || 20,
                content: node.content || ''
            }));
            
            const relationships = (mindmapData.relationships || []).filter(rel => 
                rel.source_id && rel.target_id &&
                nodes.some(n => n.id === rel.source_id) &&
                nodes.some(n => n.id === rel.target_id)
            );

            if (nodes.length === 0) {
                document.getElementById('mindmapStatus').textContent = 'No valid nodes found in mind map';
                return;
            }
        } catch (error) {
            console.error('Error rendering mind map:', error);
            const statusEl = document.getElementById('mindmapStatus');
            if (statusEl) {
                statusEl.textContent = `Error: ${error.message}`;
            }
            return;
        }

        const svg = d3.select('#mindmapSvg');
        const container = document.getElementById('mindmapContainer');
        const width = container.clientWidth;
        const height = 600;

        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(relationships).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(25));

        // Create links
        const links = mindmapG.selectAll('.mindmap-link')
            .data(relationships)
            .enter().append('line')
            .attr('class', 'mindmap-link');

        // Create nodes
        const nodeGroup = mindmapG.selectAll('.mindmap-node-group')
            .data(nodes)
            .enter().append('g')
            .attr('class', 'mindmap-node-group')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        // Add circles for nodes
            nodeGroup.append('circle')
                .attr('class', d => `mindmap-node ${d.type || 'general'}`)
                .attr('r', d => Math.min(Math.max(d.size || 20, 15), 30))
            .on('mouseover', showTooltip)
            .on('mouseout', hideTooltip)
            .on('click', showNodeDetails);

        // Add text labels
        nodeGroup.append('text')
            .attr('class', 'mindmap-text')
            .attr('dy', '.35em')
            .text(d => {
                const name = d.name || 'Unknown';
                return name.length > 12 ? name.substring(0, 12) + '...' : name;
            });

        // Update positions on simulation tick
        simulation.on('tick', () => {
            links
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeGroup
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });

        // Drag functions
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        // Tooltip functions
        function showTooltip(event, d) {
            const tooltip = d3.select('body').append('div')
                .attr('class', 'mindmap-tooltip visible')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .html(`
                    <strong>${d.name}</strong><br/>
                    Type: ${d.type}<br/>
                    ${d.content ? d.content.substring(0, 200) + '...' : 'No content available'}
                `);
        }

        function hideTooltip() {
            d3.selectAll('.mindmap-tooltip').remove();
        }

        function showNodeDetails(event, d) {
            console.log("Node clicked:", d);
            alert(`Node: ${d.name}\n\nContent: ${d.content}`);
        }
    };

    const refreshMindMap = async () => {
        try {
            const statusDiv = document.getElementById('mindmapStatus');
            statusDiv.style.display = 'block';
            statusDiv.textContent = 'Refreshing mindmap...';
            
            // Completely stop and remove existing simulation
            if (simulation) {
                simulation.stop();
                simulation.nodes([]);
                simulation = null;
            }
            
            // Clear all SVG elements and reset DOM
            const svg = d3.select('#mindmapSvg');
            svg.selectAll('*').remove();
            svg.attr('transform', null);
            
            // Reset zoom state
            svg.call(zoom.transform, d3.zoomIdentity);
            
            // Reinitialize visualization components
            initializeMindMap();
            
            // Reload fresh data from server
            await loadMindMapData();
            
            // Force full redraw
            svg.attr('viewBox', `0 0 ${svg.attr('width')} ${svg.attr('height')}`);
            
        } catch (error) {
            console.error('Refresh error:', error);
            statusDiv.textContent = `Refresh failed: ${error.message}`;
        }
    };

    // Initialize mind map only if visualization container exists
    if (document.getElementById('mindmapContainer')) {
        setTimeout(() => {
            try {
                initializeMindMap();
                loadMindMapData();
            } catch (error) {
                console.error('Mindmap initialization failed:', error);
                const statusEl = document.getElementById('mindmapStatus');
                if (statusEl) {
                    statusEl.innerHTML = `❌ Mindmap failed to initialize: ${error.message}`;
                }
            }
        }, 1000);
    }

    // --- Initial Load ---
    if (autoRefreshToggle.checked) {
        startHealthMonitoring(); // Start with auto-refresh if enabled
    } else {
        checkHealth(); // Just do one check
    }
    fetchAndRenderCaches();
    fetchLlmConfig();

    // --- Vector Mind Map Functionality ---
    class VectorMindMapVisualizer {
        constructor() {
            this.apiBase = '/api';
            this.mindmapServiceBase = '/mindmap/api/v1';
            this.currentMindMap = null;
            this.availableCaches = [];
            this.svg = null;
            this.simulation = null;
            this.nodes = [];
            this.links = [];
            this.tooltip = null;
            this.transform = d3.zoomIdentity;
            
            this.init();
        }
        
        init() {
            this.setupEventListeners();
            this.loadAvailableCaches();
            
            // Auto-generate mind map if enabled and caches are available
            setTimeout(() => {
                const autoCheckbox = document.getElementById('auto-mindmap-enabled');
                if (autoCheckbox?.checked) {
                    this.createVectorMindMap();
                }
            }, 2000); // Give time for caches to load
        }
        
        setupEventListeners() {
            // Range input listeners
            document.getElementById('mindmap-max-concepts').addEventListener('input', (e) => {
                document.getElementById('mindmap-max-concepts-value').textContent = e.target.value;
            });
            
            // Auto-generation checkbox
            document.getElementById('auto-mindmap-enabled').addEventListener('change', (e) => {
                if (e.target.checked && this.availableCaches.length > 0) {
                    this.createVectorMindMap();
                }
            });
            
            // Note: Buttons use onclick attributes in HTML instead of event listeners
            // to avoid conflicts and ensure global function calls work properly
            console.log('✅ Mind map event listeners set up (except buttons which use onclick)');
        }
        
        async loadAvailableCaches() {
            try {
                console.log('Loading available caches from:', `${this.apiBase}/cache/list`);
                const response = await fetch(`${this.apiBase}/cache/list`);
                if (response.ok) {
                    const data = await response.json();
                    console.log('Cache response data:', data);
                    // Use the correct property name from the API response
                    this.availableCaches = data.available_caches || [];
                    const statusMessage = this.availableCaches.length > 0 
                        ? `Found ${this.availableCaches.length} available caches - Ready for mind map generation`
                        : 'No caches available - Upload documents first';
                    this.updateStatus(statusMessage, this.availableCaches.length > 0 ? 'success' : 'info');
                    console.log('Available caches loaded:', this.availableCaches);
                } else {
                    console.log('Cache response not ok:', response.status, response.statusText);
                    this.availableCaches = [];
                    this.updateStatus('No caches available from RAG service', 'error');
                }
            } catch (error) {
                console.warn('Failed to load caches from RAG service:', error);
                this.availableCaches = [];
                this.updateStatus('RAG service not available', 'error');
            }
        }
        
        async createVectorMindMap() {
            console.log('🚀 createVectorMindMap called');
            const createBtn = document.getElementById('create-mindmap-btn');
            const resetBtn = document.getElementById('reset-mindmap-btn');
            
            if (!createBtn) {
                console.error('❌ Create button not found in createVectorMindMap');
                return;
            }
            
            createBtn.disabled = true;
            createBtn.textContent = '🔄 Creating...';
            
            this.updateStatus('Creating vector-based mind map...', 'loading');
            console.log('📊 Status updated, starting mind map creation process');
            
            try {
                // Get settings from UI
                const focusElement = document.getElementById('mindmap-focus');
                const maxConceptsElement = document.getElementById('mindmap-max-concepts');
                
                console.log('🎛️ UI Elements found:', {
                    focus: !!focusElement,
                    maxConcepts: !!maxConceptsElement
                });
                
                if (!focusElement || !maxConceptsElement) {
                    throw new Error('Required UI elements not found');
                }
                
                const focus = focusElement.value;
                const maxConcepts = parseInt(maxConceptsElement.value);
                
                console.log('🎛️ UI Settings:', { focus, maxConcepts });
                
                // Use available caches or create demo
                let cacheHashes = [];
                if (this.availableCaches.length > 0) {
                    console.log('Available caches:', this.availableCaches);
                    cacheHashes = this.availableCaches.slice(0, 3).map(cache => cache.hash).filter(hash => hash != null);
                    console.log('Extracted cache hashes:', cacheHashes);
                    
                    if (cacheHashes.length === 0) {
                        console.warn('No valid cache hashes found, creating demo');
                        this.updateStatus('No valid document caches found. Upload documents first or view demo.', 'error');
                        this.createDemoMindMap();
                        return;
                    }
                } else {
                    console.log('No available caches, creating demo');
                    this.updateStatus('No document caches available. Upload documents first or view demo.', 'error');
                    // Create demo mind map with sample data
                    this.createDemoMindMap();
                    return;
                }
                
                const focusOnTechnical = focus === 'technical';
                const title = this.generateAutoTitle(focus, cacheHashes.length);
                
                const requestPayload = {
                    cache_hashes: cacheHashes,
                    title: title,
                    max_nodes: maxConcepts,  // Changed from max_concepts to max_nodes
                    similarity_threshold: 0.4,
                    description: `Vector-based mind map focusing on ${focus} concepts`,
                    layout: "force",
                    include_documents: true,
                    include_concepts: true,
                    include_keywords: focusOnTechnical  // Use technical focus for keywords
                };
                
                console.log('📡 Sending request to mindmap service...');
                
                // First, test if the mindmap service is reachable
                try {
                    console.log('🔍 Testing mindmap service connectivity...');
                    const healthUrl = `${this.mindmapServiceBase.replace('/api/v1', '')}/health`;
                    console.log('🔍 Health URL:', healthUrl);
                    
                    const healthResponse = await fetch(healthUrl, {
                        method: 'GET'
                    });
                    console.log('🔍 Health check response:', healthResponse.status, healthResponse.statusText);
                    
                    if (!healthResponse.ok) {
                        console.warn('⚠️ Health check failed but service responded');
                    }
                } catch (healthError) {
                    console.error('⚠️ Mindmap service health check failed:', healthError);
                    throw new Error(`Mindmap service not reachable at ${this.mindmapServiceBase}. Is it running on port 8003?`);
                }
                
                // Create vector mind map using the mindmap service
                const response = await fetch(`${this.mindmapServiceBase}/concepts/create_vector_mindmap`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestPayload)
                });
                
                console.log('📡 Response received:', response.status, response.statusText);
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('✅ Success response:', result);
                    
                    this.currentMindMap = result.mindmap;
                    
                    // Visualize the mind map
                    this.visualizeMindMap(result.mindmap, result.technical_concepts || []);
                    
                    // Handle stats - check both possible formats
                    const stats = result.extraction_stats || result.stats || {};
                    const statsMessage = stats.total_concepts 
                        ? `✅ Mind map created! ${stats.total_concepts} concepts, ${stats.technical_concepts || 0} technical`
                        : `✅ Mind map created successfully!`;
                        
                    this.updateStatus(statsMessage, 'success');
                    
                    // Show legend
                    const legendEl = document.getElementById('mindmap-legend');
                    if (legendEl) legendEl.style.display = 'block';
                    
                } else {
                    const errorText = await response.text();
                    console.error('❌ HTTP Error:', response.status, errorText);
                    let errorData;
                    try {
                        errorData = JSON.parse(errorText);
                    } catch {
                        errorData = { detail: errorText };
                    }
                    throw new Error(errorData.detail || `HTTP ${response.status}: ${errorText}`);
                }
                
            } catch (error) {
                console.error('❌ Full error details:', error);
                console.error('❌ Error stack:', error.stack);
                console.error('❌ Error message:', error.message);
                
                let errorMessage = error.message;
                if (error.name === 'TypeError' && error.message.includes('fetch')) {
                    errorMessage = 'Network error: Cannot connect to mindmap service. Is it running?';
                } else if (error.message.includes('Mindmap service not reachable')) {
                    errorMessage = error.message;
                }
                
                this.updateStatus(`❌ Error: ${errorMessage}`, 'error');
                
                // Fallback to demo
                console.log('🎭 Creating demo mind map as fallback...');
                this.createDemoMindMap();
            } finally {
                createBtn.disabled = false;
                createBtn.textContent = '🧠 Generate Mind Map';
            }
        }
        
        createDemoMindMap() {
            // Create a demo mind map with sample technical concepts
            const demoNodes = [
                { id: 'ai', name: 'Artificial Intelligence', type: 'technical', size: 30 },
                { id: 'ml', name: 'Machine Learning', type: 'technical', size: 25 },
                { id: 'nlp', name: 'Natural Language Processing', type: 'technical', size: 20 },
                { id: 'neural', name: 'Neural Networks', type: 'technical', size: 22 },
                { id: 'data', name: 'Data Science', type: 'general', size: 18 },
                { id: 'python', name: 'Python', type: 'technical', size: 15 },
                { id: 'api', name: 'API', type: 'technical', size: 16 },
                { id: 'vector', name: 'Vector Database', type: 'technical', size: 19 }
            ];
            
            const demoLinks = [
                { source: 'ai', target: 'ml', strength: 0.9 },
                { source: 'ml', target: 'neural', strength: 0.8 },
                { source: 'ml', target: 'nlp', strength: 0.7 },
                { source: 'data', target: 'ml', strength: 0.6 },
                { source: 'python', target: 'ml', strength: 0.5 },
                { source: 'api', target: 'vector', strength: 0.4 },
                { source: 'nlp', target: 'ai', strength: 0.7 }
            ];
            
            const demoMindMap = {
                id: 'demo',
                title: 'Demo Technical Concept Map',
                nodes: demoNodes,
                relationships: demoLinks.map(link => ({
                    source_id: link.source,
                    target_id: link.target,
                    weight: link.strength
                }))
            };
            
            this.visualizeMindMap(demoMindMap);
            this.updateStatus('✅ Demo mind map created! Upload documents for real analysis', 'success');
            document.getElementById('mindmap-legend').style.display = 'block';
        }
        
        visualizeMindMap(mindmap, technicalConcepts = []) {
            try {
                // Validate input data
                if (!mindmap || !mindmap.nodes || !Array.isArray(mindmap.nodes)) {
                    throw new Error('Invalid mindmap data');
                }

                // Clear existing visualization
                d3.select('#mindmap-visualization').selectAll('*').remove();
                
                // Hide loading message
                const loadingEl = document.getElementById('mindmap-loading');
                if (loadingEl) loadingEl.style.display = 'none';
            } catch (error) {
                console.error('Visualization initialization error:', error);
                this.updateStatus(`Initialization failed: ${error.message}`, 'error');
                return;
            }
            
            // Set up SVG
            const container = document.getElementById('mindmap-visualization');
            const width = container.clientWidth;
            const height = container.clientHeight || 500;
            
            this.svg = d3.select('#mindmap-visualization')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            // Set up zoom
            const zoom = d3.zoom()
                .scaleExtent([0.3, 3])
                .on('zoom', (event) => {
                    this.svg.select('.zoom-group').attr('transform', event.transform);
                });
            
            this.svg.call(zoom);
            
            const g = this.svg.append('g').attr('class', 'zoom-group');
            
            // Prepare data
            this.nodes = mindmap.nodes.map(node => {
                if (!node.id || !node.name) {
                    console.warn('Invalid node format:', node);
                    return null;
                }
                return {
                    id: node.id,
                    name: node.name,
                    type: node.type || 'general',
                    size: node.size || 15,
                    technical: technicalConcepts.includes(node.name) || node.type === 'technical'
                };
            }).filter(Boolean);

            if (this.nodes.length === 0) {
                throw new Error('No valid nodes to visualize');
            }
            
            this.links = (mindmap.relationships || []).map(rel => {
                const sourceNode = this.nodes.find(n => n.id === rel.source_id);
                const targetNode = this.nodes.find(n => n.id === rel.target_id);
                
                if (!sourceNode || !targetNode) {
                    console.warn('Invalid relationship:', rel);
                    return null;
                }
                
                return {
                    source: sourceNode,
                    target: targetNode,
                    strength: rel.weight || 0.5
                };
            }).filter(Boolean);
            
            // Set up force simulation
            this.simulation = d3.forceSimulation(this.nodes)
                .force('link', d3.forceLink(this.links).id(d => d.id).distance(80))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => d.size + 5));
            
            // Add links
            const link = g.append('g')
                .selectAll('line')
                .data(this.links)
                .enter().append('line')
                .attr('class', 'mindmap-link')
                .style('stroke-width', d => Math.sqrt(d.strength * 5));
            
            // Add nodes
            const node = g.append('g')
                .selectAll('circle')
                .data(this.nodes)
                .enter().append('circle')
                .attr('class', d => `mindmap-node ${d.technical ? 'technical' : 'general'}`)
                .attr('r', d => d.size)
                .call(d3.drag()
                    .on('start', this.dragstarted.bind(this))
                    .on('drag', this.dragged.bind(this))
                    .on('end', this.dragended.bind(this)));
            
            // Add labels
            const label = g.append('g')
                .selectAll('text')
                .data(this.nodes)
                .enter().append('text')
                .attr('class', 'mindmap-label')
                .text(d => d.name.length > 15 ? d.name.substring(0, 15) + '...' : d.name);
            
            // Add tooltip
            this.tooltip = d3.select('body').append('div')
                .attr('class', 'mindmap-tooltip')
                .style('display', 'none');
            
            // Add hover events
            node.on('mouseover', (event, d) => {
                this.tooltip
                    .style('display', 'block')
                    .html(`<strong>${d.name}</strong><br/>Type: ${d.technical ? 'Technical' : 'General'}<br/>Size: ${d.size}`)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px');
            })
            .on('mouseout', () => {
                this.tooltip.style('display', 'none');
            })
            .on('click', (event, d) => {
                this.showNodeDetails(event, d);
            });
            
            // Update positions on simulation tick
            this.simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                label
                    .attr('x', d => d.x)
                    .attr('y', d => d.y + 4);
            });
        }
        
        dragstarted(event, d) {
            if (!event.active) this.simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        dragended(event, d) {
            if (!event.active) this.simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        async showNodeDetails(event, d) {
            console.log("Node clicked:", d);
            const modal = document.getElementById('content-detail-modal');
            const titleEl = document.getElementById('modal-title');
            const rawEl = document.getElementById('modal-raw-content');
            const markdownEl = document.getElementById('modal-markdown-content');

            titleEl.textContent = `Mind Map Node: ${d.name}`;
            rawEl.innerHTML = '<p>Loading relevant documents...</p>';
            markdownEl.innerHTML = '<p>Loading relevant documents...</p>';
            
            // Show raw content by default for node details
            showModalTab('raw');
            modal.style.display = 'block';

            // Always fetch the latest list of all available caches to search across everything
            await this.loadAvailableCaches();
            const allCacheHashes = this.availableCaches.map(c => c.hash);

            if (!allCacheHashes || allCacheHashes.length === 0) {
                rawEl.innerHTML = '<p>Error: No available caches to search.</p>';
                markdownEl.innerHTML = '<p>Error: No available caches to search.</p>';
                return;
            }

            try {
                const response = await fetch(`${this.apiBase}/mindmap/documents_for_concept`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        concept: d.name,
                        cache_hashes: allCacheHashes, // Use all available hashes
                        top_k: 5
                    })
                });

                if (!response.ok) {
                    // For any server error (like 500), display a user-friendly "no data" message.
                    console.error(`Server error fetching documents for concept '${d.name}': ${response.status}`);
                    rawEl.innerHTML = '<p>No relevant documents found for this concept.</p>';
                    markdownEl.innerHTML = '<p>No relevant documents found for this concept.</p>';
                    return;
                }

                const data = await response.json();

                if (data.documents && data.documents.length > 0) {
                    let html = '<h3>Relevant Documents:</h3>';
                    let rawContent = `Relevant Documents for "${d.name}":\n\n`;
                    
                    data.documents.forEach((doc, index) => {
                        const source = doc.metadata.source_url || 'Unknown source';
                        const score = doc.score ? ` (Relevance: ${doc.score.toFixed(2)})` : '';
                        
                        html += `
                            <div class="document">
                                <h4><a href="${source}" target="_blank">${source}</a>${score}</h4>
                                <div class="document-content">${doc.content}</div>
                            </div>
                        `;
                        
                        rawContent += `--- Document ${index + 1}: ${source}${score} ---\n`;
                        rawContent += doc.content + '\n\n';
                    });
                    
                    rawEl.innerHTML = `<pre>${rawContent}</pre>`;
                    markdownEl.innerHTML = html;
                } else {
                    rawEl.innerHTML = '<p>No relevant documents found for this concept.</p>';
                    markdownEl.innerHTML = '<p>No relevant documents found for this concept.</p>';
                }

            } catch (error) {
                console.error('Error fetching node details:', error);
                // Also treat network errors as "no data" from the user's perspective.
                rawEl.innerHTML = '<p>No relevant documents found for this concept.</p>';
                markdownEl.innerHTML = '<p>No relevant documents found for this concept.</p>';
            }
        }
        
        resetMindMap() {
            if (this.svg) {
                const container = document.getElementById('mindmap-visualization');
                const width = container.clientWidth;
                const height = container.clientHeight || 500;
                
                const resetTransform = d3.zoomIdentity;
                this.svg.transition().duration(750).call(
                    d3.zoom().transform,
                    resetTransform
                );
                
                // Restart simulation
                if (this.simulation) {
                    this.simulation.alpha(1).restart();
                }
            }
        }
        
        generateAutoTitle(focus, cacheCount) {
            const focusText = {
                'technical': 'Technical',
                'general': 'General',
                'mixed': 'Mixed'
            };
            
            return `${focusText[focus]} Concept Map (${cacheCount} sources)`;
        }
        
        updateStatus(message, type = 'info') {
            const statusEl = document.getElementById('mindmap-status');
            statusEl.textContent = message;
            statusEl.className = `status-message ${type}`;
        }

        // Method to trigger auto-generation from external calls
        async tryAutoGeneration() {
            try {
                const autoCheckbox = document.getElementById('auto-mindmap-enabled');
                if (autoCheckbox?.checked) {
                    console.log('Triggering auto-generation from external call');
                    // Refresh caches first, then create mind map
                    await this.loadAvailableCaches();
                    console.log('Caches reloaded, available caches:', this.availableCaches.length);
                    
                    if (this.availableCaches.length > 0) {
                        // Add a small delay to ensure backend cache is fully updated
                        setTimeout(() => {
                            this.createVectorMindMap();
                        }, 500);
                    } else {
                        console.log('No caches available for mind map generation after reload');
                    }
                }
            } catch (error) {
                console.error('Error in tryAutoGeneration:', error);
            }
        }

        // Method to manually refresh caches (can be called from outside)
        async refreshCaches() {
            console.log('Manually refreshing mind map caches');
            this.updateStatus('Refreshing available caches...', 'loading');
            await this.loadAvailableCaches();
            console.log(`Cache refresh complete: ${this.availableCaches.length} caches available`);
            return this.availableCaches.length;
        }

        // Method to manually trigger mind map creation (for debugging)
        async forceCreateMindMap() {
            console.log('Force creating mind map...');
            await this.loadAvailableCaches();
            if (this.availableCaches.length > 0) {
                this.createVectorMindMap();
            } else {
                console.log('No caches available for mind map creation');
                this.createDemoMindMap();
            }
        }
    }

    // Initialize Vector Mind Map
    let vectorMindMapVisualizer;

    // Initialize mind map visualizer immediately (DOM is already ready since this is inside DOMContentLoaded)
    try {
        if (document.getElementById('mindmap-visualization')) {
            console.log('🚀 Initializing VectorMindMapVisualizer');
            vectorMindMapVisualizer = new VectorMindMapVisualizer();
            window.vectorMindMapVisualizer = vectorMindMapVisualizer;
            console.log('✅ Mind map visualizer initialized successfully');
        } else {
            console.warn('⚠️ Mind map visualization element not found');
        }
    } catch (error) {
        console.error('❌ Mind map initialization failed:', error);
    }

    // Global functions for HTML onclick handlers
    window.createVectorMindMap = function() {
        console.log('🎯 Global createVectorMindMap called');
        console.log('🔍 Visualizer available:', !!window.vectorMindMapVisualizer);
        
        if (window.vectorMindMapVisualizer) {
            console.log('✅ Calling visualizer.createVectorMindMap()');
            try {
                window.vectorMindMapVisualizer.createVectorMindMap();
            } catch (error) {
                console.error('❌ Error calling createVectorMindMap:', error);
            }
        } else {
            console.error('❌ Vector mind map visualizer not initialized');
            // Try to initialize it now
            try {
                if (document.getElementById('mindmap-visualization')) {
                    console.log('🔄 Attempting to initialize visualizer now...');
                    window.vectorMindMapVisualizer = new VectorMindMapVisualizer();
                    window.vectorMindMapVisualizer.createVectorMindMap();
                } else {
                    console.error('❌ Mind map visualization element not found!');
                }
            } catch (error) {
                console.error('❌ Failed to initialize visualizer:', error);
            }
        }
    };

    window.resetMindMap = function() {
        console.log('🎯 Global resetMindMap called');
        console.log('🔍 Visualizer available:', !!window.vectorMindMapVisualizer);
        if (window.vectorMindMapVisualizer) {
            window.vectorMindMapVisualizer.resetMindMap();
        } else {
            console.error('❌ Vector mind map visualizer not initialized');
        }
    };

    // Debug functions for browser console
    window.refreshMindMapCaches = function() {
        console.log('refreshMindMapCaches called, visualizer:', window.vectorMindMapVisualizer);
        if (window.vectorMindMapVisualizer) {
            return window.vectorMindMapVisualizer.refreshCaches();
        } else {
            console.error('Vector mind map visualizer not initialized');
            return Promise.resolve(0);
        }
    };

    window.forceCreateMindMap = function() {
        console.log('forceCreateMindMap called, visualizer:', window.vectorMindMapVisualizer);
        if (window.vectorMindMapVisualizer) {
            window.vectorMindMapVisualizer.forceCreateMindMap();
        } else {
            console.error('Vector mind map visualizer not initialized');
        }
    };

    // --- Modal Logic ---
    const modal = document.getElementById('content-detail-modal');
    const closeButton = document.querySelector('.close-button');

    if(modal && closeButton) {
        closeButton.onclick = function() {
            modal.style.display = "none";
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
    }
});
