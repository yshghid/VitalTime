<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Search Section -->
    <div class="bg-blue-100 p-6">
      <div class="max-w-4xl mx-auto">
        <h1 class="text-2xl font-bold text-gray-800 mb-4">Patient Search</h1>
        <div class="flex flex-col sm:flex-row gap-4 items-end">
          <div class="flex-1">
            <label for="patientSelect" class="block text-sm font-medium text-gray-700 mb-2">
              Select Patient
            </label>
            <select
              id="patientSelect"
              v-model="selectedPatientName"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select a patient...</option>
              <option
                v-for="patient in patientList"
                :key="patient.patient_name + patient.timestamp"
                :value="patient.patient_name"
              >
                {{ patient.patient_name }}
              </option>
            </select>
          </div>
          <div class="sm:flex-none">
            <button
              @click="searchPatient"
              class="w-full sm:w-auto px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="!selectedPatientName"
            >
              Search
            </button>
          </div>
          <div class="sm:flex-none">
            <button
              @click="loadAllPatients"
              class="w-full sm:w-auto px-6 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors"
            >
              Show All
            </button>
          </div>
        </div>
        <div class="mt-3 text-sm text-gray-600">
          Total: {{ totalCount }} patients
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="max-w-6xl mx-auto p-6">
      <div class="text-center py-12">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
        <div class="text-gray-500 text-sm mt-4">Loading patients...</div>
      </div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="max-w-6xl mx-auto p-6">
      <div class="bg-red-50 border border-red-200 rounded-lg p-6">
        <div class="flex">
          <div class="text-red-400">
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">Error loading patients</h3>
            <div class="mt-2 text-sm text-red-700">{{ error }}</div>
            <button
              @click="loadAllPatients"
              class="mt-3 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Patient Cards Section -->
    <div v-else class="max-w-6xl mx-auto p-6">
      <!-- Empty State -->
      <div v-if="displayedPatients.length === 0" class="text-center py-12">
        <div class="text-gray-400 text-lg mb-2">No patients found</div>
        <div class="text-gray-500 text-sm">Try adjusting your search criteria</div>
      </div>

      <!-- Patient Cards Grid -->
      <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          v-for="patient in displayedPatients"
          :key="patient.patient_name + patient.timestamp"
          @click="goToPatientDetail(patient)"
          class="bg-white rounded-lg border border-gray-200 shadow-md p-6 hover:shadow-lg transition-shadow cursor-pointer"
        >
          <div class="space-y-3">
            <div class="flex justify-between items-start">
              <h3 class="text-lg font-semibold text-gray-900">{{ patient.patient_name }}</h3>
              <span
                :class="getSeverityBadgeClass(patient.cur_predicted)"
                class="px-2 py-1 rounded-full text-xs font-medium"
              >
                {{ patient.cur_predicted }}
              </span>
            </div>
            <div class="text-sm text-gray-600 space-y-1">
              <p><span class="font-medium">기록시간:</span> {{ formatTimestamp(patient.timestamp) }}</p>
              <p><span class="font-medium">현재 중증도:</span> {{ patient.cur_news }}</p>
              <p><span class="font-medium">8시간 뒤 예측 중증도:</span> {{ patient.cur_predicted }}</p>
            </div>
            <div class="mt-4 pt-4 border-t border-gray-100">
              <div class="flex items-center justify-between">
                <span class="text-sm text-gray-500">중증도 점수</span>
                <div class="flex items-center">
                  <div class="w-16 bg-gray-200 rounded-full h-2 mr-2">
                    <div
                      :class="getSeverityBarClass(patient.cur_predicted)"
                      class="h-2 rounded-full transition-all duration-300"
                      :style="`width: ${(patient.cur_predicted / 10) * 100}%`"
                    ></div>
                  </div>
                  <span class="text-sm font-medium text-gray-700">
                    {{ patient.cur_predicted }}/10
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

// Configure axios
const api = axios.create({
  baseURL: (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) || 'http://localhost:8001'
})

// Reactive state
const patientList = ref([])
const displayedPatients = ref([])
const selectedPatientName = ref('')
const loading = ref(false)
const error = ref('')
const totalCount = ref(0)

// API function to fetch patient info
const fetchPatientInfo = async () => {
  try {
    loading.value = true
    error.value = ''

    // Use fixed timestamp as requested
    const timestamp = new Date('2025-01-02T12:00:00').toISOString()

    const response = await api.get('/api/get-patient-info', {
      params: { timestamp }
    })

    const data = response.data
    patientList.value = data.patients || []
    displayedPatients.value = data.patients || []
    totalCount.value = data.total_count || 0

  } catch (err) {
    console.error('Error fetching patient info:', err)
    error.value = err.response?.data?.detail || err.message || 'Failed to load patient data. Please check if the backend server is running.'
    patientList.value = []
    displayedPatients.value = []
    totalCount.value = 0
  } finally {
    loading.value = false
  }
}

// Methods
const loadAllPatients = async () => {
  selectedPatientName.value = ''
  await fetchPatientInfo()
}

const searchPatient = () => {
  if (!selectedPatientName.value) {
    displayedPatients.value = patientList.value
    return
  }

  displayedPatients.value = patientList.value.filter(
    patient => patient.patient_name === selectedPatientName.value
  )
}

// Utility function to format timestamp
const formatTimestamp = (timestamp) => {
  if (!timestamp) return '-'

  try {
    const date = new Date(timestamp)
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')
    const hours = String(date.getHours()).padStart(2, '0')
    const minutes = String(date.getMinutes()).padStart(2, '0')
    const seconds = String(date.getSeconds()).padStart(2, '0')

    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`
  } catch (err) {
    return timestamp
  }
}

// Utility functions for styling based on cur_predicted score
const getSeverityBadgeClass = (score) => {
  if (score >= 7) return 'bg-red-100 text-red-800'
  if (score >= 4) return 'bg-yellow-100 text-yellow-800'
  return 'bg-green-100 text-green-800'
}

const getSeverityBarClass = (score) => {
  if (score >= 7) return 'bg-red-500'
  if (score >= 4) return 'bg-yellow-500'
  return 'bg-green-500'
}

// Navigate to patient detail page with name and timestamp
const goToPatientDetail = (patient) => {
  router.push({
    name: 'PatientDetail',
    query: {
      name: patient.patient_name,
      timestamp: patient.timestamp
    }
  })
}

// Load data on component mount
onMounted(() => {
  fetchPatientInfo()
})
</script>