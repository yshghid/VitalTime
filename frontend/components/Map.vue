<template>
  <div class="hospital-map-app">
    <div class="hospital-map-container">
      <div id="hospital-map" class="hospital-map"></div>
    </div>
    
    <div class="hospital-sidebar">
      <div class="hospital-header">
        <h1 class="hospital-title">병원 검색</h1>
        <div class="hospital-distance-buttons">
          <button 
            v-for="distance in distances" 
            :key="distance"
            :class="['hospital-distance-btn', { active: selectedDistance === distance }]"
            @click="selectDistance(distance)"
          >
            {{ distance }}km
          </button>
        </div>
      </div>
      
      <div class="hospital-list">
        <div v-if="loading" class="hospital-loading">
          병원을 검색 중입니다...
        </div>
        <div v-else-if="filteredHospitals.length === 0" class="hospital-no-hospitals">
          선택한 반경 내에 병원이 없습니다.
        </div>
        <div v-else>
          <div 
            v-for="hospital in filteredHospitals" 
            :key="hospital.id"
            class="hospital-item"
            @click="toggleHospital(hospital.id)"
          >
            <input 
              type="checkbox" 
              class="hospital-checkbox"
              :checked="selectedHospitals.includes(hospital.id)"
              @click.stop="toggleHospital(hospital.id)"
            >
            <div class="hospital-info">
              <div class="hospital-name">{{ hospital.name }}</div>
              <div class="hospital-address">{{ hospital.address }}</div>
              <div class="hospital-distance">{{ hospital.distance }}km</div>
              <div class="hospital-phone">{{ hospital.phone }}</div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="hospital-stats">
        선택된 병원: <span class="hospital-selected-count">{{ selectedHospitals.length }}</span>개 / 
        전체: {{ filteredHospitals.length }}개
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Map',
  data() {
    return {
      map: null,
      userMarker: null,
      circle: null,
      circles: [], // 모든 원을 관리하는 배열
      hospitalMarkers: [],
      userLocation: null,
      selectedDistance: 3,
      distances: [3, 5, 10, 20],
      selectedHospitals: [],
      loading: false,
      hospitals: [
        {
          id: 1,
          name: "서울대학교병원",
          address: "서울특별시 종로구 대학로 101",
          lat: 37.5665,
          lng: 126.9780,
          distance: 2.3,
          phone: "02-2072-2114"
        },
        {
          id: 2,
          name: "삼성서울병원",
          address: "서울특별시 강남구 일원로 81",
          lat: 37.4881,
          lng: 127.0856,
          distance: 1.8,
          phone: "02-3410-2114"
        },
        {
          id: 3,
          name: "세브란스병원",
          address: "서울특별시 서대문구 연세로 50-1",
          lat: 37.5623,
          lng: 126.9408,
          distance: 3.2,
          phone: "02-2228-5800"
        },
        {
          id: 4,
          name: "고려대학교안암병원",
          address: "서울특별시 성북구 고려대로 73",
          lat: 37.5893,
          lng: 127.0263,
          distance: 4.1,
          phone: "02-920-5114"
        },
        {
          id: 5,
          name: "한양대학교병원",
          address: "서울특별시 성동구 왕십리로 222-1",
          lat: 37.5583,
          lng: 127.0440,
          distance: 2.7,
          phone: "02-2290-8114"
        },
        {
          id: 6,
          name: "서울아산병원",
          address: "서울특별시 송파구 올림픽로43길 88",
          lat: 37.5267,
          lng: 127.1108,
          distance: 6.8,
          phone: "02-3010-3114"
        },
        {
          id: 7,
          name: "중앙대학교병원",
          address: "서울특별시 동작구 흑석로 102",
          lat: 37.5068,
          lng: 126.9590,
          distance: 5.2,
          phone: "02-6299-1114"
        },
        {
          id: 8,
          name: "경희대학교병원",
          address: "서울특별시 동대문구 경희대로 23",
          lat: 37.5946,
          lng: 127.0526,
          distance: 3.9,
          phone: "02-958-8114"
        }
      ]
    }
  },
  computed: {
    filteredHospitals() {
      return this.hospitals.filter(hospital => 
        hospital.distance <= this.selectedDistance
      );
    }
  },
  mounted() {
    this.initMap();
  },
  methods: {
    initMap() {
      // 병원들의 중심점 계산
      this.calculateHospitalCenter();
      this.setupMap();
    },
    calculateHospitalCenter() {
      // 병원들의 평균 좌표 계산
      const totalLat = this.hospitals.reduce((sum, hospital) => sum + hospital.lat, 0);
      const totalLng = this.hospitals.reduce((sum, hospital) => sum + hospital.lng, 0);
      
      this.userLocation = {
        lat: totalLat / this.hospitals.length,
        lng: totalLng / this.hospitals.length
      };
    },
    setupMap() {
      this.map = new google.maps.Map(document.getElementById("hospital-map"), {
        center: this.userLocation,
        zoom: 14,
      });

      // 중심점 마커 (병원들의 중심)
      this.userMarker = new google.maps.Marker({
        position: this.userLocation,
        map: this.map,
        title: "병원 중심점",
        icon: "http://maps.google.com/mapfiles/ms/icons/blue-dot.png",
      });

      this.updateCircle();
      this.addHospitalMarkers();
    },
    selectDistance(distance) {
      this.selectedDistance = distance;
      this.updateCircle();
      this.addHospitalMarkers();
    },
    updateCircle() {
      // 모든 기존 원들을 완전히 제거
      this.circles.forEach(circle => {
        if (circle) {
          circle.setMap(null);
        }
      });
      this.circles = [];
      
      // 현재 원도 제거
      if (this.circle) {
        this.circle.setMap(null);
        this.circle = null;
      }

      // 새로운 원 생성
      this.circle = new google.maps.Circle({
        map: this.map,
        center: this.userLocation,
        radius: this.selectedDistance * 1000, // km를 m로 변환
        fillColor: "#000000",
        fillOpacity: 0.1,
        strokeColor: "#000000",
        strokeOpacity: 0.6,
        strokeWeight: 2,
      });
      
      // 새로 생성된 원을 배열에 추가
      this.circles.push(this.circle);
    },
    addHospitalMarkers() {
      // 기존 마커들 제거
      if (this.hospitalMarkers) {
        this.hospitalMarkers.forEach(marker => marker.setMap(null));
      }
      this.hospitalMarkers = [];

      // 선택한 범위 내의 병원들만 마커로 표시
      this.filteredHospitals.forEach(hospital => {
        const marker = new google.maps.Marker({
          position: { lat: hospital.lat, lng: hospital.lng },
          map: this.map,
          title: hospital.name,
          icon: {
            url: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
            scaledSize: new google.maps.Size(30, 30)
          }
        });

        const infoWindow = new google.maps.InfoWindow({
          content: `
            <div style="padding: 10px;">
              <h3 style="margin: 0 0 5px 0; color: #333;">${hospital.name}</h3>
              <p style="margin: 0; color: #666; font-size: 14px;">${hospital.address}</p>
              <p style="margin: 5px 0 0 0; color: #888; font-size: 12px;">거리: ${hospital.distance}km</p>
            </div>
          `
        });

        marker.addListener('click', () => {
          infoWindow.open(this.map, marker);
        });

        this.hospitalMarkers.push(marker);
      });
    },
    toggleHospital(hospitalId) {
      const index = this.selectedHospitals.indexOf(hospitalId);
      if (index > -1) {
        // 이미 선택된 병원이면 선택 해제
        this.selectedHospitals.splice(index, 1);
      } else {
        // 선택되지 않은 병원이면 선택 추가
        this.selectedHospitals.push(hospitalId);
      }
    }
  }
}
</script>

<style>
/* 전역 스타일 - 다른 프로젝트에서 사용 가능 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #f5f5f5;
  color: #333;
}

.hospital-map-app {
  display: flex;
  height: 100vh;
}

.hospital-map-container {
  flex: 1;
  position: relative;
}

.hospital-map {
  height: 100%;
  width: 100%;
}

.hospital-sidebar {
  width: 400px;
  background: white;
  border-left: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  box-shadow: -2px 0 10px rgba(0,0,0,0.1);
}

.hospital-header {
  padding: 20px;
  border-bottom: 1px solid #e0e0e0;
  background: #fafafa;
}

.hospital-title {
  font-size: 24px;
  font-weight: 600;
  color: #333;
  margin-bottom: 15px;
}

.hospital-distance-buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.hospital-distance-btn {
  padding: 8px 16px;
  border: 2px solid #333;
  background: white;
  color: #333;
  cursor: pointer;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.hospital-distance-btn:hover {
  background: #f0f0f0;
}

.hospital-distance-btn.active {
  background: #333;
  color: white;
}

.hospital-list {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.hospital-item {
  display: flex;
  align-items: center;
  padding: 15px 0;
  border-bottom: 1px solid #f0f0f0;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.hospital-item:hover {
  background-color: #f9f9f9;
}

.hospital-checkbox {
  margin-right: 15px;
  width: 18px;
  height: 18px;
  accent-color: #333;
}

.hospital-info {
  flex: 1;
}

.hospital-name {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.hospital-address {
  font-size: 14px;
  color: #666;
  margin-bottom: 2px;
}

.hospital-distance {
  font-size: 12px;
  color: #888;
}

.hospital-phone {
  font-size: 12px;
  color: #888;
  margin-top: 2px;
}

.hospital-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #666;
}

.hospital-no-hospitals {
  text-align: center;
  padding: 40px 20px;
  color: #666;
}

.hospital-stats {
  padding: 15px 20px;
  background: #f9f9f9;
  border-top: 1px solid #e0e0e0;
  font-size: 14px;
  color: #666;
}

.hospital-selected-count {
  font-weight: 600;
  color: #333;
}
</style>
