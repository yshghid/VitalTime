<template>
  <div class="min-h-screen bg-gray-100 p-4 md:p-8">
    <div class="max-w-7xl mx-auto">
      <div class="flex flex-col lg:flex-row gap-6">
        <!-- Left Column (1/3) - Scrollable -->
        <div class="w-full lg:w-1/3">
          <div class="bg-white border border-gray-200 shadow-md rounded-lg p-6 max-h-screen overflow-y-auto">
            <h2 class="text-xl font-bold mb-6 text-gray-800">Patient Information</h2>

            <!-- Basic Info Section -->
            <div class="space-y-3 mb-6">
              <div class="flex justify-between py-2">
                <span class="text-gray-600">Patient ID</span>
                <span class="text-gray-900 font-medium">{{ patientInfo.id || '-' }}</span>
              </div>

              <div class="flex justify-between py-2">
                <span class="text-gray-600">Patient Name</span>
                <span class="text-gray-900 font-medium">{{ patientInfo.name || '-' }}</span>
              </div>

              <div class="flex justify-between py-2">
                <span class="text-gray-600">Admission Date</span>
                <span class="text-gray-900 font-medium">{{ patientInfo.admitDate || '-' }}</span>
              </div>
            </div>

            <!-- Blood/Lab Indicators Section -->
            <div class="border-t border-gray-200 pt-4 mb-6">
              <h3 class="text-sm font-bold text-gray-700 mb-3">Blood/Lab Indicators</h3>
              <div class="space-y-4">
                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">Creatinine</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.creatinine }} mg/dL</span>
                  </div>
                  <div class="text-xs text-gray-500">(kidney function)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">Hemoglobin</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.hemoglobin }} g/dL</span>
                  </div>
                  <div class="text-xs text-gray-500">(anemia, oxygen)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">LDH</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.ldh }} U/L</span>
                  </div>
                  <div class="text-xs text-gray-500">(tissue damage)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">Lymphocytes</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.lymphocytes }}%</span>
                  </div>
                  <div class="text-xs text-gray-500">(immune status)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">Neutrophils</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.neutrophils }}%</span>
                  </div>
                  <div class="text-xs text-gray-500">(infection)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">Platelet Count</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.platelet_count?.toLocaleString() }} /μL</span>
                  </div>
                  <div class="text-xs text-gray-500">(clotting)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">WBC Count</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.wbc_count?.toLocaleString() }} /μL</span>
                  </div>
                  <div class="text-xs text-gray-500">(infection)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">hs-CRP</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.hs_crp }} mg/L</span>
                  </div>
                  <div class="text-xs text-gray-500">(inflammation)</div>
                </div>

                <div>
                  <div class="flex justify-between py-1">
                    <span class="text-gray-600">D-Dimer</span>
                    <span class="text-gray-900 font-medium">{{ patientInfo.d_dimer }} μg/mL</span>
                  </div>
                  <div class="text-xs text-gray-500">(thrombosis)</div>
                </div>
              </div>
            </div>

            <button
              @click="predictSeverity"
              :disabled="isLoading"
              class="w-full mt-6 px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
            >
              <svg v-if="isLoading" class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ isLoading ? 'Predicting...' : 'Predict Severity' }}
            </button>
          </div>
        </div>

        <!-- Right Column (2/3) -->
        <div class="w-full lg:w-2/3 space-y-6">
          <!-- Severity Result Display -->
          <div class="bg-white border border-gray-200 shadow-md rounded-lg p-8">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">Severity Result</h3>
            <div v-if="!isPredicted" class="flex items-center justify-center min-h-[120px] border-2 border-dashed border-gray-300 rounded-lg bg-gray-50">
              <div class="text-gray-400 text-lg">
                Severity Score Display Area
              </div>
            </div>
            <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center p-6 border border-gray-300 rounded-lg">
              <!-- Left: Real-time Heart Rate -->
              <div class="text-center md:text-left">
                <div class="text-xs text-gray-500 mb-1">현재 심박수</div>
                <div class="text-6xl font-bold text-red-500 flex items-center justify-center md:justify-start">
                  <svg class="w-12 h-12 mr-2 animate-pulse" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clip-rule="evenodd"></path></svg>
                  {{ currentHeartRate }}
                </div>
                <div class="text-lg text-gray-600 -mt-2 ml-14">bpm</div>
              </div>
              <!-- Right: Severity Score -->
              <div class="text-center md:text-right border-t md:border-t-0 md:border-l border-gray-200 pt-4 md:pt-0 md:pl-6">
                <div class="text-xs text-gray-500 mb-1">8시간 후 예측 중증도</div>
                <div class="text-6xl font-bold" :class="getSeverityColor(severityResult.score)">
                  {{ severityResult.score }}
                </div>
                 <div class="text-lg text-gray-600">Score</div>
              </div>
            </div>
          </div>

          <!-- Nearby Hospitals Map -->
          <div class="bg-white border border-gray-200 shadow-md rounded-lg p-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">Nearby Hospitals Map</h3>
            <div class="bg-gray-100 rounded-lg p-6 h-[400px] relative flex items-center justify-center">
              <div v-if="hospitals.length === 0" class="text-gray-500">
                Nearby Hospitals Map
              </div>
              <div v-else class="w-full h-full grid grid-cols-3 md:grid-cols-5 gap-4 place-items-center">
                <div
                  v-for="hospital in hospitals"
                  :key="hospital.id"
                  @click="toggleHospitalSelection(hospital.id)"
                  class="cursor-pointer p-4 rounded-lg border-2 transition-all relative group"
                  :class="selectedHospitals.includes(hospital.id)
                    ? 'bg-blue-100 border-blue-500'
                    : 'bg-white border-gray-300 hover:border-blue-400'"
                >
                  <div class="text-4xl">📍</div>
                  <!-- Tooltip -->
                  <div class="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                    {{ hospital.name }}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Patient Transfer Button -->
          <button
            @click="createTransferManual"
            :disabled="selectedHospitals.length === 0"
            class="w-full px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:bg-green-400 disabled:cursor-not-allowed transition-colors"
          >
            Create Patient Transfer Manual
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import axios from 'axios'

const route = useRoute()
const router = useRouter()

// State
const patientInfo = ref({})
const severityResult = ref({
  score: null,
  predictedAt: null
})
const isPredicted = ref(false)
const isLoading = ref(false)
const hospitals = ref([])
const selectedHospitals = ref([])
const currentHeartRate = ref(78)
let heartRateInterval = null

// Get severity color based on score
const getSeverityColor = (score) => {
  if (score >= 7) return 'text-red-600'
  if (score >= 4) return 'text-yellow-600'
  return 'text-green-600'
}

// Fetch patient details
const getPatientDetail = async (patientId) => {
  try {
    // API call
    const response = await axios.get(`/api/getPatientDetail?id=${patientId}`)
    patientInfo.value = response.data
  } catch (error) {
    console.error('Error fetching patient details:', error)
    alert('Failed to fetch patient details')
  }
}

// Predict severity
const predictSeverity = async () => {
  isLoading.value = true
  try {
    // Mock API call
    // const response = await axios.post('/api/predictSeverity', {
    //   patientId: patientInfo.value.id,
    //   medicalData: patientInfo.value
    // })
    // severityResult.value = response.data

    // Mock data for testing
    await new Promise(resolve => setTimeout(resolve, 1500))

    severityResult.value = {
      score: Math.floor(Math.random() * 10) + 1, // Random score 1-10
      predictedAt: '2025-09-30 14:32:15'
    }
    isPredicted.value = true

    // Fetch nearby hospitals after prediction
    await getNearbyHospitals()
  } catch (error) {
    console.error('Error predicting severity:', error)
    alert('Failed to predict severity')
  } finally {
    isLoading.value = false
  }
}

// Fetch nearby hospitals
const getNearbyHospitals = async () => {
  try {
    // Mock API call
    // const response = await axios.get('/api/getNearbyHospitals')
    // hospitals.value = response.data

    // Mock data for testing
    hospitals.value = [
      { id: 1, name: '서울대병원', lat: 37.5, lng: 127.0 },
      { id: 2, name: '삼성서울병원', lat: 37.48, lng: 127.08 },
      { id: 3, name: '아산병원', lat: 37.52, lng: 127.11 },
      { id: 4, name: '연세대병원', lat: 37.56, lng: 126.94 },
      { id: 5, name: '서울성모병원', lat: 37.50, lng: 127.00 }
    ]
  } catch (error) {
    console.error('Error fetching nearby hospitals:', error)
    alert('Failed to fetch nearby hospitals')
  }
}

// Toggle hospital selection
const toggleHospitalSelection = (hospitalId) => {
  const index = selectedHospitals.value.indexOf(hospitalId)
  if (index > -1) {
    selectedHospitals.value.splice(index, 1)
  } else {
    selectedHospitals.value.push(hospitalId)
  }
}

// Create transfer manual
const createTransferManual = () => {
  if (selectedHospitals.value.length === 0) {
    alert('Please select at least one hospital')
    return
  }

  const selectedHospitalData = hospitals.value.filter(h =>
    selectedHospitals.value.includes(h.id)
  )

  alert(`Creating transfer manual for:\n${selectedHospitalData.map(h => h.name).join('\n')}`)

  // Navigate to next page with data
  // router.push({
  //   name: 'TransferManual',
  //   query: {
  //     patientId: patientInfo.value.id,
  //     hospitals: selectedHospitals.value.join(',')
  //   }
  // })
}

// Initialize
onMounted(() => {
  const patientId = route.query.id || 'P001'
  getPatientDetail(patientId)

  // Start heart rate simulation
  heartRateInterval = setInterval(() => {
    const baseRate = 75
    const fluctuation = Math.floor(Math.random() * 11) - 5 // -5 to +5
    currentHeartRate.value = baseRate + fluctuation
  }, 1000)
})

onUnmounted(() => {
  // Clean up interval on component unmount
  if (heartRateInterval) {
    clearInterval(heartRateInterval)
  }
})
</script>

<style scoped>
@keyframes pulse {
  50% {
    transform: scale(1.1);
  }
}
.animate-pulse {
  animation: pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
</style>