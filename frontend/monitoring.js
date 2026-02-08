
const { createApp, ref, onMounted } = Vue

const MonitoringDashboard = {
  setup() {
    const apiLogs = ref([])
    const mlLogs = ref([])
    const loading = ref(false)
    const error = ref(null)

    const fetchLogs = async () => {
      loading.value = true
      error.value = null
      try {
        const baseUrl = (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) || 'http://localhost:8001'
        const apiRes = await fetch(`${baseUrl}/api/monitoring/api`)
        if (!apiRes.ok) throw new Error(`API logs: ${apiRes.statusText}`)
        const apiData = await apiRes.json()
        apiLogs.value = apiData.map(log => JSON.parse(log.replace(/'/g, '"')))

        const mlRes = await fetch(`${baseUrl}/api/monitoring/ml`)
        if (!mlRes.ok) throw new Error(`ML logs: ${mlRes.statusText}`)
        const mlData = await mlRes.json()
        mlLogs.value = mlData.map(log => JSON.parse(log))

      } catch (e) {
        error.value = e.message
      } finally {
        loading.value = false
      }
    }

    onMounted(fetchLogs)

    const goBack = () => {
      window.location.href = '/'
    }

    return {
      apiLogs,
      mlLogs,
      loading,
      error,
      fetchLogs,
      goBack
    }
  },
  template: `
    <div class="min-h-screen bg-gray-100">
      <div class="container mx-auto p-4">
        <div class="flex items-center justify-between mb-4">
          <h1 class="text-3xl font-bold">AIOps Monitoring Dashboard</h1>
          <div class="flex gap-2">
            <button @click="goBack" class="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors">
              ← 뒤로가기
            </button>
            <button @click="fetchLogs" :disabled="loading" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400">
              {{ loading ? 'Refreshing...' : 'Refresh' }}
            </button>
          </div>
        </div>

        <div v-if="error" class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
          <strong class="font-bold">Error:</strong>
          <span class="block sm:inline">{{ error }}</span>
        </div>

        <!-- API Monitoring Logs -->
        <div class="bg-white shadow-md rounded-lg p-4 mb-6">
          <h2 class="text-2xl font-semibold mb-3">API Monitoring Logs</h2>
          <div class="overflow-x-auto">
            <table class="min-w-full bg-white">
              <thead class="bg-gray-200">
                <tr>
                  <th class="py-2 px-4 border-b">Path</th>
                  <th class="py-2 px-4 border-b">Method</th>
                  <th class="py-2 px-4 border-b">Status Code</th>
                  <th class="py-2 px-4 border-b">Process Time (ms)</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(log, index) in apiLogs" :key="index" class="hover:bg-gray-100">
                  <td class="py-2 px-4 border-b">{{ log.path }}</td>
                  <td class="py-2 px-4 border-b">{{ log.method }}</td>
                  <td class="py-2 px-4 border-b">{{ log.status_code }}</td>
                  <td class="py-2 px-4 border-b">{{ log.process_time_ms }}</td>
                </tr>
                <tr v-if="apiLogs.length === 0">
                    <td colspan="4" class="text-center py-4">No API logs found.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- ML Monitoring Logs -->
        <div class="bg-white shadow-md rounded-lg p-4">
          <h2 class="text-2xl font-semibold mb-3">ML Model Training Logs</h2>
          <div class="overflow-x-auto">
            <table class="min-w-full bg-white">
              <thead class="bg-gray-200">
                <tr>
                  <th class="py-2 px-4 border-b">Timestamp</th>
                  <th class="py-2 px-4 border-b">Event</th>
                  <th class="py-2 px-4 border-b">MSE</th>
                  <th class="py-2 px-4 border-b">MAE</th>
                  <th class="py-2 px-4 border-b">R²</th>
                  <th class="py-2 px-4 border-b">Training Time (s)</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(log, index) in mlLogs" :key="index" class="hover:bg-gray-100">
                  <td class="py-2 px-4 border-b">{{ new Date(log.timestamp).toLocaleString() }}</td>
                  <td class="py-2 px-4 border-b">{{ log.event }}</td>
                  <td class="py-2 px-4 border-b">{{ log.model_info?.evaluation?.mse?.toFixed(4) || 'N/A' }}</td>
                  <td class="py-2 px-4 border-b">{{ log.model_info?.evaluation?.mae?.toFixed(4) || 'N/A' }}</td>
                  <td class="py-2 px-4 border-b">{{ log.model_info?.evaluation?.r2?.toFixed(4) || 'N/A' }}</td>
                  <td class="py-2 px-4 border-b">{{ log.model_info?.training_time_seconds?.toFixed(2) || 'N/A' }}</td>
                </tr>
                 <tr v-if="mlLogs.length === 0">
                    <td colspan="6" class="text-center py-4">No ML logs found.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  `
}

createApp(MonitoringDashboard).mount('#app')
