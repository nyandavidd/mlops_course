let heatmap = null;
let HeatmapModule = null;
let map = null;
let placemark = null;

function updateYearValue(value) {
    document.getElementById('yearValue').textContent = value;
    updateHeatmap(value);
}

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371e3;
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lon2 - lon1) * Math.PI / 180;

    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    return R * c;
}

function interpolatePoints(originalPoints, bounds, zoom) {
    if (zoom < 14) return originalPoints;
    
    const interpolatedPoints = [...originalPoints];
    const gridSpacing = 200;
    
    const minLat = bounds[0][0];
    const maxLat = bounds[1][0];
    const minLon = bounds[0][1];
    const maxLon = bounds[1][1];
    
    const latStep = gridSpacing / 111320;
    const lonStep = gridSpacing / (111320 * Math.cos((minLat + maxLat) * Math.PI / 360));
    
    for (let lat = minLat; lat <= maxLat; lat += latStep) {
        for (let lon = minLon; lon <= maxLon; lon += lonStep) {
            let totalWeight = 0;
            let weightedSum = 0;
            let hasNearbyPoint = false;
            
            for (const point of originalPoints) {
                const distance = calculateDistance(lat, lon, point[1], point[2]);
                if (distance < 2000) {
                    hasNearbyPoint = true;
                    const weight = 1 / (distance * distance);
                    totalWeight += weight;
                    weightedSum += point[0] * weight;
                }
            }
            
            if (hasNearbyPoint && totalWeight > 0) {
                const interpolatedValue = weightedSum / totalWeight;
                interpolatedPoints.push([interpolatedValue, lat, lon, originalPoints[0][3]]);
            }
        }
    }
    
    return interpolatedPoints;
}

async function updateHeatmap(year) {
    if (!HeatmapModule) {
        console.error('Heatmap module not loaded');
        return;
    }

    try {
        const response = await fetch(`/get_heatmap_data?year=${year}`);
        if (!response.ok) {
            throw new Error('Ошибка при получении данных тепловой карты');
        }
        
        const data = await response.json();
        const points = data.points;
        
        if (points.length === 0) {
            console.warn('Нет данных для выбранного года');
            return;
        }

        let minPrice = Infinity;
        let maxPrice = -Infinity;
        
        for (const point of points) {
            const price = parseFloat(point[0]);
            if (!isNaN(price)) {
                minPrice = Math.min(minPrice, price);
                maxPrice = Math.max(maxPrice, price);
            }
        }
        
        const bounds = map.getBounds();
        const zoom = map.getZoom();
        const interpolatedPoints = interpolatePoints(points, bounds, zoom);
        
        const heatmapPoints = [];
        const logMinPrice = Math.log(minPrice);
        const logMaxPrice = Math.log(maxPrice);
        const logRange = logMaxPrice - logMinPrice;
        
        for (const point of interpolatedPoints) {
            const price = parseFloat(point[0]);
            const lat = parseFloat(point[1]);
            const lon = parseFloat(point[2]);
            
            if (!isNaN(price) && !isNaN(lat) && !isNaN(lon)) {
                const normalizedPrice = (Math.log(price) - logMinPrice) / logRange;
                heatmapPoints.push({
                    type: 'Feature',
                    geometry: {
                        type: 'Point',
                        coordinates: [lat, lon]
                    },
                    properties: {
                        weight: normalizedPrice
                    }
                });
            }
        }

        if (heatmap) {
            heatmap.setMap(null);
        }
        
        const radius = Math.max(20, Math.min(100, 50 + (zoom - 10) * 5));
        
        heatmap = new HeatmapModule({
            type: 'FeatureCollection',
            features: heatmapPoints
        }, {
            radius: radius,
            dissipating: true,
            opacity: 0.4,
            intensityOfMidpoint: 0.05,
            gradient: {
                0.0: 'rgb(0, 0, 255)',    // Синий для дешевых за кв.м
                0.1: 'rgb(0, 255, 0)',    // Зеленый
                0.3: 'rgb(255, 255, 0)',  // Желтый
                0.5: 'rgb(255, 165, 0)',  // Оранжевый
                0.8: 'rgb(255, 69, 0)',   // Красно-оранжевый
                1.0: 'rgb(255, 0, 0)'     // Красный для дорогих за кв.м
            }
        });

        heatmap.setMap(map);

    } catch (error) {
        console.error('Ошибка при обновлении тепловой карты:', error);
    }
}

function validateForm() {
    const lat = document.getElementById('geo_lat').value;
    const lon = document.getElementById('geo_lon').value;
    if (!lat || !lon) {
        alert('Пожалуйста, выберите местоположение на карте');
        return false;
    }
    return true;
}

async function submitForm() {
    if (!validateForm()) {
        return;
    }
    const form = document.querySelector('form');
    const formData = new FormData(form);
    const button = document.getElementById('predictButton');
    const resultDiv = document.getElementById('predictionResult');

    try {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Загрузка...';
        const response = await fetch('/', {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Ошибка сети');
        }

        const data = await response.json();
        
        const formattedPrice = new Intl.NumberFormat('ru-RU').format(Math.round(data.prediction));
        resultDiv.innerHTML = `<h2 class="prediction-result">Предсказанная цена: ${formattedPrice} рублей</h2>`;

    } catch (error) {
        console.error('Ошибка:', error);
        resultDiv.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
    } finally {
        button.disabled = false;
        button.innerHTML = 'Предсказать цену';
    }
}

function createPlacemark(coords) {
    return new ymaps.Placemark(coords, {}, {
        preset: 'islands#violetDotIconWithCaption',
        draggable: true
    });
}

function getAddress(coords) {
    document.getElementById('geo_lat').value = coords[0];
    document.getElementById('geo_lon').value = coords[1];
}

function initMap(Heatmap) {
    console.log('Initializing map...');
    
    const defaultCoords = [59.93863, 30.31413];
    const savedLat = document.getElementById('geo_lat').value;
    const savedLon = document.getElementById('geo_lon').value;
    let initialCoords;
    
    if (savedLat && savedLon) {
        initialCoords = [parseFloat(savedLat), parseFloat(savedLon)];
        console.log('Using coordinates from form:', initialCoords);
    } else {
        const savedCoords = localStorage.getItem('lastMarkerPosition');
        if (savedCoords) {
            initialCoords = JSON.parse(savedCoords);
            console.log('Using saved coordinates from localStorage:', initialCoords);
        } else {
            initialCoords = defaultCoords;
            console.log('Using default coordinates:', initialCoords);
        }
    }
    
    map = new ymaps.Map("map", {
        center: initialCoords,
        zoom: 11,
        controls: ['zoomControl', 'fullscreenControl']
    });
    placemark = createPlacemark(initialCoords);
    map.geoObjects.add(placemark);
    getAddress(initialCoords);
    const currentYear = document.getElementById('year').value;
    updateHeatmap(currentYear);
    
    map.events.add('click', function(e) {
        const coords = e.get('coords');
        if (placemark) {
            map.geoObjects.remove(placemark);
        }
        
        placemark = createPlacemark(coords);
        map.geoObjects.add(placemark);
        localStorage.setItem('lastMarkerPosition', JSON.stringify(coords));
        console.log('Saved marker position:', coords);

        placemark.events.add('dragend', function() {
            const newCoords = placemark.geometry.getCoordinates();
            getAddress(newCoords);
            localStorage.setItem('lastMarkerPosition', JSON.stringify(newCoords));
            console.log('Saved marker position after drag:', newCoords);
        });
        
        getAddress(coords);
    });
    
    map.events.add('boundschange', function() {
        const currentYear = document.getElementById('year').value;
        updateHeatmap(currentYear);
    });
}

function initApp() {
    document.getElementById('predictButton').addEventListener('click', submitForm);
    ymaps.ready(function() {
        ymaps.modules.require(['Heatmap'], function(Heatmap) {
            console.log('Heatmap module loaded');
            HeatmapModule = Heatmap;
            initMap(Heatmap);
        });
    });
}
document.addEventListener('DOMContentLoaded', initApp); 