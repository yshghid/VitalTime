<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header Section -->
    <div class="bg-blue-100 p-6">
      <div class="max-w-7xl mx-auto">
        <div class="flex items-center justify-between mb-4">
          <h1 class="text-2xl font-bold text-gray-800">환자 전원 정보</h1>
          <button
            @click="goBack"
            class="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors"
          >
            ← 뒤로가기
          </button>
        </div>
        <div class="text-sm text-gray-600">
          선택된 환자의 상세 정보와 AI 생성 환자 전원 의뢰서를 확인하세요.
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center items-center py-20">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      <div class="ml-4 text-gray-600">보고서를 생성하고 있습니다...</div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="max-w-7xl mx-auto p-6">
      <div class="bg-red-50 border border-red-200 rounded-lg p-6">
        <div class="flex">
          <div class="text-red-400">
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-red-800">오류가 발생했습니다</h3>
            <div class="mt-2 text-sm text-red-700">{{ error }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <div v-else class="max-w-7xl mx-auto p-6">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <!-- Left Column: Patient & Hospital Info -->
        <div class="space-y-6">
          <!-- Patient Information Card -->
          <div class="bg-white rounded-lg border border-gray-200 shadow-md">
            <div class="px-6 py-4 border-b border-gray-200">
              <h3 class="text-lg font-semibold text-gray-900 flex items-center">
                <svg class="h-5 w-5 text-blue-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                환자 정보
              </h3>
            </div>
            <div class="px-6 py-4 space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700">환자명</label>
                <div class="mt-1 text-lg font-semibold text-gray-900">{{ patientInfo?.patient_name || '-' }}</div>
              </div>
              <div>
                <label class="block text-sm font-medium text-gray-700">환자 ID</label>
                <div class="mt-1 text-gray-900">{{ patientInfo?.patient_id || '-' }}</div>
              </div>
              <div>
                <label class="block text-sm font-medium text-gray-700">중증도</label>
                <div class="mt-1">
                  <span
                    :class="getSeverityBadgeClass(patientInfo?.severity)"
                    class="px-3 py-1 rounded-full text-sm font-medium"
                  >
                    {{ patientInfo?.severity || '-' }}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- Hospital Information Card -->
          <div class="bg-white rounded-lg border border-gray-200 shadow-md">
            <div class="px-6 py-4 border-b border-gray-200">
              <h3 class="text-lg font-semibold text-gray-900 flex items-center">
                <svg class="h-5 w-5 text-green-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-4m-5 0H3m2 0h4M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                </svg>
                이송 병원 정보
              </h3>
            </div>
            <div class="px-6 py-4 space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700">병원명</label>
                <div class="mt-1 text-lg font-semibold text-gray-900">{{ hospitalInfo?.name || '-' }}</div>
              </div>
              <div>
                <label class="block text-sm font-medium text-gray-700">주소</label>
                <div class="mt-1 text-gray-900">{{ hospitalInfo?.address || '-' }}</div>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium text-gray-700">거리</label>
                  <div class="mt-1 text-gray-900">{{ hospitalInfo?.distance || '-' }}km</div>
                </div>
                <div>
                  <label class="block text-sm font-medium text-gray-700">연락처</label>
                  <div class="mt-1 text-gray-900">{{ hospitalInfo?.phone || '-' }}</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Clinical Data Card -->
          <div class="bg-white rounded-lg border border-gray-200 shadow-md">
            <div class="px-6 py-4 border-b border-gray-200">
              <h3 class="text-lg font-semibold text-gray-900 flex items-center">
                <svg class="h-5 w-5 text-purple-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                최신 검사 수치
              </h3>
            </div>
            <div class="px-6 py-4 space-y-4">
              <div class="grid grid-cols-1 gap-4">
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                  <span class="text-sm font-medium text-gray-700">D-Dimer</span>
                  <span class="text-sm font-semibold text-gray-900">
                    {{ clinicalData?.d_dimer || '-' }} ng/mL
                  </span>
                </div>
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                  <span class="text-sm font-medium text-gray-700">LDH</span>
                  <span class="text-sm font-semibold text-gray-900">
                    {{ clinicalData?.ldh || '-' }} U/L
                  </span>
                </div>
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                  <span class="text-sm font-medium text-gray-700">Creatinine</span>
                  <span class="text-sm font-semibold text-gray-900">
                    {{ clinicalData?.creatinine || '-' }} mg/dL
                  </span>
                </div>
                <div class="flex justify-between items-center py-2">
                  <span class="text-sm font-medium text-gray-700">검사 시점</span>
                  <span class="text-sm font-semibold text-gray-900">
                    {{ clinicalData?.timepoint || '-' }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: AI Report -->
        <div>
          <div class="bg-white rounded-lg border border-gray-200 shadow-md h-full">
            <div class="px-6 py-4 border-b border-gray-200">
              <h3 class="text-lg font-semibold text-gray-900 flex items-center">
                <svg class="h-5 w-5 text-orange-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                AI 생성 환자 전원 의뢰서
                <span class="ml-2 text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded-full">
                  자동 생성됨
                </span>
              </h3>
            </div>
            <div class="px-6 py-4">
              <div
                class="bg-gray-50 rounded-lg p-4 min-h-[600px] font-mono text-sm leading-relaxed whitespace-pre-line overflow-auto border"
                style="font-family: 'Courier New', monospace;"
              >
                {{ aiReport?.report_content || '보고서를 생성하고 있습니다...' }}
              </div>
              <div v-if="aiReport?.generated_at" class="mt-3 text-xs text-gray-500 text-right">
                생성 시간: {{ formatDateTime(aiReport.generated_at) }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="mt-8 flex justify-center space-x-4">
        <button
          @click="printReport"
          class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
        >
          <svg class="h-4 w-4 inline mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
          </svg>
          보고서 출력
        </button>
        <button
          @click="downloadReport"
          class="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-colors"
        >
          <svg class="h-4 w-4 inline mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          다운로드
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

// Props for receiving data from previous page
const props = defineProps({
  patientId: {
    type: Number,
    required: true

  },
  hospitalInfo: {
    type: Object,
    required: true
  }
})

// Reactive state
const loading = ref(true)
const error = ref('')
const patientInfo = ref(null)
const hospitalInfo = ref(props.hospitalInfo)
const clinicalData = ref(null)
const aiReport = ref(null)

// API call to get patient report
const fetchPatientReport = async () => {
  try {
    loading.value = true
    error.value = ''

    const response = await axios.post('/api/page3/patient-report', {
      patient_id: props.patientId,
      hospital_info: props.hospitalInfo
    })

    const data = response.data
    patientInfo.value = data.patient_info
    hospitalInfo.value = data.hospital_info
    clinicalData.value = data.clinical_data
    aiReport.value = data.ai_report

  } catch (err) {
    console.error('Error fetching patient report:', err)
    error.value = err.response?.data?.detail || '데이터를 불러오는 중 오류가 발생했습니다.'

    // Mock data for development/testing
    loadMockData()
  } finally {
    loading.value = false
  }
}

// Mock data for development
const loadMockData = () => {
  patientInfo.value = {
    patient_id: props.patientId,
    patient_name: "홍길동",
    severity: "중증"
  }

  clinicalData.value = {
    d_dimer: 2.5,
    ldh: 450,
    creatinine: 1.8,
    timepoint: 24
  }

  aiReport.value = {
    report_content: `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    환자 전원 의뢰서
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

■ 환자 기본 정보
 • 성명: ${patientInfo.value.patient_name}
 • 환자번호: ${patientInfo.value.patient_id}
 • 중증도 분류: ${patientInfo.value.severity}

■ 이송 의료기관
 • 기관명: ${hospitalInfo.value.name}
 • 소재지: ${hospitalInfo.value.address}
 • 연락처: ${hospitalInfo.value.phone}

■ 현재 상태 및 검사 소견
 • 주요 검사 결과:
   - D-Dimer: ${clinicalData.value.d_dimer} ng/mL
   - LDH: ${clinicalData.value.ldh} U/L
   - Creatinine: ${clinicalData.value.creatinine} mg/dL

■ 전원 사유 및 임상적 판단
환자의 D-Dimer 상승(${clinicalData.value.d_dimer} ng/mL)과 LDH 증가(${clinicalData.value.ldh} U/L),
Creatinine 상승(${clinicalData.value.creatinine} mg/dL)이 관찰되어 혈전색전증 및
신기능 악화가 의심됩니다. 전문적인 중환자 치료가 필요하여 상급 의료기관으로의
전원을 권고합니다.

■ 특이사항 및 주의사항
이송 중 활력징후 모니터링이 필수이며, 신기능 저하로 인한 약물 용량 조절이
필요합니다. 혈전색전증 예방을 위한 항응고요법 고려가 권장됩니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`,
    generated_at: new Date().toISOString()
  }
}

// Utility functions
const getSeverityBadgeClass = (severity) => {
  switch (severity) {
    case '중증':
      return 'bg-red-100 text-red-800'
    case '중등도':
      return 'bg-yellow-100 text-yellow-800'
    case '경증':
      return 'bg-green-100 text-green-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}

const formatDateTime = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleString('ko-KR')
}

const goBack = () => {
  // Navigate back to previous page
  window.history.back()
}

const printReport = () => {
  window.print()
}

const downloadReport = () => {
  const content = aiReport.value?.report_content || ''
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `전원의뢰서_${patientInfo.value?.patient_name || 'unknown'}_${new Date().toISOString().split('T')[0]}.txt`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

// Load data on component mount
onMounted(() => {
  fetchPatientReport()
})
</script>

<style scoped>
@media print {
  .no-print {
    display: none !important;
  }
}
</style>