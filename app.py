import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import base64
import io
import pickle
import numpy as np
import datetime

# --- Helper functions for data loading and preprocessing ---
def load_model():
    """Loads the pre-trained heart disease prediction model."""
    try:
        with open("heart_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: The model file 'heart_model.pkl' was not found.")
        return None

# --- Translation and Theme Dictionaries ---
LANGUAGES = {
    "English": {
        "title": "❤️ Heart Disease Risk Predictor",
        "subtitle": "Predict heart disease risk using Machine Learning.",
        "get_started": "Get Started",
        "login_title": "Login / Sign Up",
        "name": "Name",
        "email": "Email",
        "password": "Password",
        "proceed": "Proceed",
        "login_success": "Login successful! You can now access the dashboard.",
        "login_error": "Please fill in all the details.",
        "dashboard_welcome": "Welcome, {username}!",
        "dashboard_subtitle": "Your health journey begins here.",
        "check_risk": "Check Risk",
        "check_risk_desc": "Input your clinical data for an instant risk assessment.",
        "view_reports": "View Reports",
        "view_reports_desc": "Review past predictions and download reports.",
        "health_tips": "Health Tips",
        "health_tips_desc": "Discover tips for a heart-healthy lifestyle.",
        "predict_title": "Predict Your Risk",
        "predict_subtitle": "Please fill in the correct information.",
        "personal_info": "Personal Information",
        "age": "Age",
        "sex": "Sex",
        "male": "Male",
        "female": "Female",
        "clinical_data": "Clinical Data",
        "cp": "Chest Pain Type",
        "cp_options": ['Typical Angina (0)', 'Atypical Angina (1)', 'Non-anginal Pain (2)', 'Asymptomatic (3)'],
        "trestbps": "Resting Blood Pressure (trestbps)",
        "chol": "Cholesterol (chol)",
        "fbs": "Fasting Blood Sugar > 120 mg/dl",
        "fbs_options": ['No (0)', 'Yes (1)'],
        "restecg": "Resting ECG Results",
        "restecg_options": ['Normal (0)', 'ST-T wave abnormality (1)', 'Left ventricular hypertrophy (2)'],
        "thalach": "Maximum Heart Rate Achieved (thalach)",
        "exang": "Exercise-induced Angina",
        "exang_options": ['No (0)', 'Yes (1)'],
        "oldpeak": "ST Depression (oldpeak)",
        "slope": "Slope of ST Segment",
        "slope_options": ['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'],
        "ca": "Number of Major Vessels (0-3)",
        "thal": "Thalassemia",
        "thal_options": ['Normal (0)', 'Fixed defect (1)', 'Reversible defect (2)'],
        "predict_button": "Predict Risk",
        "predicting": "Predicting...",
        "result_title": "Prediction Result",
        "high_risk": "High Risk Detected!",
        "high_risk_msg": "Based on the data, there is a high risk of heart disease. Please consult a medical professional.",
        "low_risk": "Low Risk!",
        "low_risk_msg": "The model predicts a low risk. Maintain your healthy lifestyle!",
        "confidence": "Confidence Score:",
        "download_report": "Download Report",
        "report_header": "Heart Disease Risk Prediction Report",
        "report_pred": "Prediction:",
        "report_conf": "Confidence:",
        "reports_title": "Prediction Reports",
        "reports_empty": "There are no reports yet. They will appear here after you make a prediction.",
        "reports_back": "Go back to Predict Page",
        "tips_title": "Health Tips",
        "tips_subtitle": "Here are some tips to keep your heart healthy:",
        "diet_tips": "Dietary Tips",
        "diet_tip1_title": "Balanced Diet",
        "diet_tip1_desc": "• Focus on a diet rich in **fruits, vegetables, and whole grains**. These provide essential vitamins, minerals, and fiber.",
        "diet_tip2_title": "Healthy Fats",
        "diet_tip2_desc": "• Reduce intake of **saturated fats** (found in red meat, butter) and opt for **healthy fats** like olive oil, avocados, and nuts.",
        "diet_tip3_title": "Sodium and Sugar",
        "diet_tip3_desc": "• Limit your salt intake to manage blood pressure. Be mindful of hidden sugars in processed foods.",
        "exercise_tips": "Exercise Routines",
        "exercise_tip1_title": "Aerobic Activity",
        "exercise_tip1_desc": "• Aim for at least 150 minutes of **moderate-intensity aerobic activity** per week. This includes brisk walking, cycling, or swimming.",
        "exercise_tip2_title": "Strength and Flexibility",
        "exercise_tip2_desc": "• Incorporate **strength training** exercises twice a week to build muscle. Don't forget flexibility exercises like stretching or yoga.",
        "exercise_tip3_title": "Stay Active",
        "exercise_tip3_desc": "• Take short walks throughout the day to break up long periods of sitting. Small movements add up!",
        "stress_tips": "Stress Management",
        "stress_tip1_title": "Mindfulness",
        "stress_tip1_desc": "• Practice **meditation, yoga, or mindfulness** to reduce stress. These activities can lower your heart rate and blood pressure.",
        "stress_tip2_title": "Quality Sleep",
        "stress_tip2_desc": "• Get adequate sleep (**7-8 hours per night**) for overall well-being. Poor sleep is linked to higher risk of heart disease.",
        "stress_tip3_title": "Connect with Others",
        "stress_tip3_desc": "• Spend time on hobbies and with loved ones to relax and find joy. A strong social network is great for heart health.",
        "settings_title": "Settings",
        "app_customization": "App Customization",
        "theme": "Theme",
        "language": "Language",
        "lang_info": "Language changed to {lang}. (Note: Not all text is fully translated for this demonstration).",
        "navigation": "Navigation",
        "dashboard": "Dashboard",
        "predict": "Predict",
        "reports": "Reports",
        "tips": "Tips",
        "settings": "Settings",
        "logout": "Logout",
        "login_prompt": "Please log in to navigate the app.",
        "app_footer": "Made by Amna",
        "font_settings": "Font Settings",
        "font_size": "Body Font Size",
        "data_management": "Data Management",
        "clear_history": "Clear history on logout",
        "reset_app": "Reset Application",
        "reset_info": "This will clear all data and restart the app.",
        "patient_name_label": "Patient Name:"
    },
    "Hindi": {
        "title": "❤️ हृदय रोग जोखिम भविष्यवक्ता",
        "subtitle": "मशीन लर्निंग का उपयोग करके हृदय रोग के जोखिम की भविष्यवाणी करें।",
        "get_started": "शुरू करें",
        "login_title": "लॉगिन / साइन अप करें",
        "name": "नाम",
        "email": "ईमेल",
        "password": "पासवर्ड",
        "proceed": "आगे बढ़ें",
        "login_success": "लॉगिन सफल रहा! अब आप डैशबोर्ड तक पहुंच सकते हैं।",
        "login_error": "कृपया सभी विवरण भरें।",
        "dashboard_welcome": "आपका स्वागत है, {username}!",
        "dashboard_subtitle": "आपकी स्वास्थ्य यात्रा यहाँ से शुरू होती है।",
        "check_risk": "जोखिम की जाँच करें",
        "check_risk_desc": "त्वरित जोखिम मूल्यांकन के लिए अपना नैदानिक ​​डेटा दर्ज करें।",
        "view_reports": "रिपोर्ट देखें",
        "view_reports_desc": "पिछली भविष्यवाणियों की समीक्षा करें और रिपोर्ट डाउनलोड करें।",
        "health_tips": "स्वास्थ्य युक्तियाँ",
        "health_tips_desc": "स्वस्थ दिल के लिए युक्तियाँ खोजें।",
        "predict_title": "अपने जोखिम का अनुमान लगाएं",
        "predict_subtitle": "कृपया सही जानकारी भरें।",
        "personal_info": "व्यक्तिगत जानकारी",
        "age": "उम्र",
        "sex": "लिंग",
        "male": "पुरुष",
        "female": "महिला",
        "clinical_data": "नैदानिक ​​डेटा",
        "cp": "सीने में दर्द का प्रकार",
        "cp_options": ['विशिष्ट एनजाइना (0)', 'असामान्य एनजाइना (1)', 'गैर-एनजाइनल दर्द (2)', 'रोगसूचक (3)'],
        "trestbps": "विश्राम रक्तचाप (trestbps)",
        "chol": "कोलेस्ट्रॉल (chol)",
        "fbs": "उपवास रक्त शर्करा > 120 मिलीग्राम/डीएल",
        "fbs_options": ['नहीं (0)', 'हाँ (1)'],
        "restecg": "विश्राम ईसीजी परिणाम",
        "restecg_options": ['सामान्य (0)', 'एसटी-टी तरंग असामान्यता (1)', 'बाएं वेंट्रिकुलर अतिवृद्धि (2)'],
        "thalach": "अधिकतम हृदय गति प्राप्त हुई (thalach)",
        "exang": "व्यायाम-प्रेरित एनजाइना",
        "exang_options": ['नहीं (0)', 'हाँ (1)'],
        "oldpeak": "एसटी अवसाद (oldpeak)",
        "slope": "एसटी खंड का ढलान",
        "slope_options": ['ऊपर की ओर (0)', 'सपाट (1)', 'नीचे की ओर (2)'],
        "ca": "प्रमुख वाहिकाओं की संख्या (0-3)",
        "thal": "थैलेसीमिया",
        "thal_options": ['सामान्य (0)', 'निश्चित दोष (1)', 'प्रतिवर्ती दोष (2)'],
        "predict_button": "जोखिम का अनुमान लगाएं",
        "predicting": "भविष्यवाणी हो रही है...",
        "result_title": "भविष्यवाणी का परिणाम",
        "high_risk": "उच्च जोखिम का पता चला!",
        "high_risk_msg": "डेटा के आधार पर, हृदय रोग का उच्च जोखिम है। कृपया एक चिकित्सा पेशेवर से परामर्श करें।",
        "low_risk": "कम जोखिम!",
        "low_risk_msg": "मॉडल कम जोखिम की भविष्यवाणी करता है। अपनी स्वस्थ जीवनशैली बनाए रखें!",
        "confidence": "आत्मविश्वास स्कोर:",
        "download_report": "रिपोर्ट डाउनलोड करें",
        "report_header": "हृदय रोग जोखिम भविष्यवाणी रिपोर्ट",
        "report_pred": "भविष्यवाणी:",
        "report_conf": "आत्मविश्वास:",
        "reports_title": "भविष्यवाणी रिपोर्ट",
        "reports_empty": "अभी तक कोई रिपोर्ट नहीं है। भविष्यवाणी करने के बाद वे यहां दिखाई देंगे।",
        "reports_back": "अनुमान पृष्ठ पर वापस जाएं",
        "tips_title": "स्वास्थ्य युक्तियाँ",
        "tips_subtitle": "अपने दिल को स्वस्थ रखने के लिए कुछ युक्तियाँ यहाँ दी गई हैं:",
        "diet_tips": "आहार संबंधी युक्तियाँ",
        "diet_tip1_title": "संतुलित आहार",
        "diet_tip1_desc": "• फल, सब्जियों और साबुत अनाज से भरपूर आहार पर ध्यान दें। ये आवश्यक विटामिन, खनिज और फाइबर प्रदान करते हैं।",
        "diet_tip2_title": "स्वस्थ वसा",
        "diet_tip2_desc": "• **संतृप्त वसा** (लाल मांस, मक्खन में पाए जाने वाले) का सेवन कम करें और जैतून का तेल, एवोकैडो और मेवे जैसे **स्वस्थ वसा** का विकल्प चुनें।",
        "diet_tip3_title": "सोडियम और चीनी",
        "diet_tip3_desc": "• रक्तचाप को नियंत्रित करने के लिए अपने नमक का सेवन सीमित करें। प्रसंस्कृत खाद्य पदार्थों में छिपी हुई शर्करा के प्रति सचेत रहें।",
        "exercise_tips": "व्यायाम दिनचर्या",
        "exercise_tip1_title": "एरोबिक गतिविधि",
        "exercise_tip1_desc": "• प्रति सप्ताह कम से कम 150 मिनट **मध्यम-तीव्रता वाली एरोबिक गतिविधि** का लक्ष्य रखें। इसमें तेज चलना, साइकिल चलाना या तैराकी शामिल है।",
        "exercise_tip2_title": "ताकत और लचीलापन",
        "exercise_tip2_desc": "• मांसपेशियों के निर्माण के लिए सप्ताह में दो बार **शक्ति प्रशिक्षण** अभ्यास शामिल करें। स्ट्रेचिंग या योग जैसे लचीलेपन वाले व्यायामों को न भूलें।",
        "exercise_tip3_title": "सक्रिय रहें",
        "exercise_tip3_desc": "• लंबे समय तक बैठने से बचने के लिए दिन भर में छोटी सैर करें। छोटे-छोटे आंदोलन बहुत फायदेमंद होते हैं!",
        "stress_tips": "तनाव प्रबंधन",
        "stress_tip1_title": "माइंडफुलनेस",
        "stress_tip1_desc": "• तनाव कम करने के लिए **ध्यान, योग या माइंडफुलनेस** का अभ्यास करें। ये गतिविधियाँ आपकी हृदय गति और रक्तचाप को कम कर सकती हैं।",
        "stress_tip2_title": "गुणवत्तापूर्ण नींद",
        "stress_tip2_desc": "• समग्र कल्याण के लिए पर्याप्त नींद लें (**प्रति रात 7-8 घंटे**)। खराब नींद हृदय रोग के उच्च जोखिम से जुड़ी है।",
        "stress_tip3_title": "दूसरों से जुड़ें",
        "stress_tip3_desc": "• आराम करने और आनंद पाने के लिए शौक और प्रियजनों के साथ समय बिताएं। एक मजबूत सामाजिक नेटवर्क दिल के स्वास्थ्य के लिए बहुत अच्छा है।",
        "settings_title": "सेटिंग्स",
        "app_customization": "ऐप अनुकूलन",
        "theme": "थीम",
        "language": "भाषा",
        "lang_info": "भाषा बदलकर {lang} हो गई। (नोट: इस प्रदर्शन के लिए सभी पाठों का पूरी तरह से अनुवाद नहीं किया गया है)।",
        "navigation": "नेविगेशन",
        "dashboard": "डैशबोर्ड",
        "predict": "अनुमान",
        "reports": "रिपोर्ट",
        "tips": "युक्तियाँ",
        "settings": "सेटिंग्स",
        "logout": "लॉग आउट",
        "login_prompt": "ऐप नेविगेट करने के लिए कृपया लॉग इन करें।",
        "app_footer": "अम्ना द्वारा बनाया गया",
        "font_settings": "फ़ॉन्ट सेटिंग्स",
        "font_size": "बॉडी फ़ॉन्ट का आकार",
        "data_management": "डेटा प्रबंधन",
        "clear_history": "लॉग आउट पर इतिहास साफ़ करें",
        "reset_app": "एप्लिकेशन रीसेट करें",
        "reset_info": "यह सभी डेटा साफ़ कर देगा और ऐप को पुनरारंभ करेगा।",
        "patient_name_label": "रोगी का नाम:"
    },
    "Spanish": {
        "title": "❤️ Predictor de Riesgo de Enfermedad Cardíaca",
        "subtitle": "Predice el riesgo de enfermedad cardíaca usando Machine Learning.",
        "get_started": "Comenzar",
        "login_title": "Iniciar Sesión / Registrarse",
        "name": "Nombre",
        "email": "Correo electrónico",
        "password": "Contraseña",
        "proceed": "Proceder",
        "login_success": "¡Inicio de sesión exitoso! Ahora puedes acceder al panel de control.",
        "login_error": "Por favor, completa todos los detalles.",
        "dashboard_welcome": "¡Bienvenido, {username}!",
        "dashboard_subtitle": "Tu viaje de salud comienza aquí.",
        "check_risk": "Verificar Riesgo",
        "check_risk_desc": "Ingresa tus datos clínicos para una evaluación de riesgo instantánea.",
        "view_reports": "Ver Informes",
        "view_reports_desc": "Revisa predicciones pasadas y descarga informes.",
        "health_tips": "Consejos de Salud",
        "health_tips_desc": "Descubre consejos para un estilo de vida saludable para el corazón.",
        "predict_title": "Predice Tu Riesgo",
        "predict_subtitle": "Por favor, completa la información correcta.",
        "personal_info": "Información Personal",
        "age": "Edad",
        "sex": "Sexo",
        "male": "Masculino",
        "female": "Femenino",
        "clinical_data": "Datos Clínicos",
        "cp": "Tipo de Dolor de Pecho",
        "cp_options": ['Angina Típica (0)', 'Angina Atípica (1)', 'Dolor no Anginal (2)', 'Asintomático (3)'],
        "trestbps": "Presión Arterial en Reposo (trestbps)",
        "chol": "Colesterol (chol)",
        "fbs": "Azúcar en la Sangre en Ayunas > 120 mg/dl",
        "fbs_options": ['No (0)', 'Sí (1)'],
        "restecg": "Resultados de ECG en Reposo",
        "restecg_options": ['Normal (0)', 'Anomalía de la onda ST-T (1)', 'Hipertrofia ventricular izquierda (2)'],
        "thalach": "Frecuencia Cardíaca Máxima Alcanzada (thalach)",
        "exang": "Angina Inducida por Ejercicio",
        "exang_options": ['No (0)', 'Sí (1)'],
        "oldpeak": "Depresión ST (oldpeak)",
        "slope": "Pendiente del Segmento ST",
        "slope_options": ['Ascendente (0)', 'Plana (1)', 'Descendente (2)'],
        "ca": "Número de Vasos Mayores (0-3)",
        "thal": "Talasemia",
        "thal_options": ['Normal (0)', 'Defecto Fijo (1)', 'Defecto Reversible (2)'],
        "predict_button": "Predecir Riesgo",
        "predicting": "Prediciendo...",
        "result_title": "Resultado de la Predicción",
        "high_risk": "¡Riesgo Alto Detectado!",
        "high_risk_msg": "Según los datos, existe un alto riesgo de enfermedad cardíaca. Por favor, consulta a un profesional médico.",
        "low_risk": "¡Riesgo Bajo!",
        "low_risk_msg": "El modelo predice un riesgo bajo. ¡Mantén tu estilo de vida saludable!",
        "confidence": "Puntuación de Confianza:",
        "download_report": "Descargar Informe",
        "report_header": "Informe de Predicción de Riesgo de Enfermedad Cardíaca",
        "report_pred": "Predicción:",
        "report_conf": "Confianza:",
        "reports_title": "Informes de Predicción",
        "reports_empty": "Aún no hay informes. Aparecerán aquí después de que hagas una predicción.",
        "reports_back": "Volver a la página de Predicción",
        "tips_title": "Consejos de Salud",
        "tips_subtitle": "Aquí tienes algunos consejos para mantener tu corazón saludable:",
        "diet_tips": "Consejos Dietéticos",
        "diet_tip1_title": "Dieta Equilibrada",
        "diet_tip1_desc": "• Concéntrate en una dieta rica en **frutas, verduras y granos enteros**. Estos proporcionan vitaminas, minerales y fibra esenciales.",
        "diet_tip2_title": "Grasas Saludables",
        "diet_tip2_desc": "• Reduce la ingesta de **grasas saturadas** (que se encuentran en la carne roja, mantequilla) y opta por **grasas saludables** como el aceite de oliva, aguacates y nueces.",
        "diet_tip3_title": "Sodio y Azúcar",
        "diet_tip3_desc": "• Limita tu ingesta de sal para controlar la presión arterial. Ten cuidado con los azúcares ocultos en los alimentos procesados.",
        "exercise_tips": "Rutinas de Ejercicio",
        "exercise_tip1_title": "Actividad Aeróbica",
        "exercise_tip1_desc": "• Aspira a al menos 150 minutos de **actividad aeróbica de intensidad moderada** por semana. Esto incluye caminar a paso ligero, andar en bicicleta o nadar.",
        "exercise_tip2_title": "Fuerza y Flexibilidad",
        "exercise_tip2_desc": "• Incorpora ejercicios de **entrenamiento de fuerza** dos veces por semana para construir músculo. No olvides los ejercicios de flexibilidad como el estiramiento o el yoga.",
        "exercise_tip3_title": "Mantente Activo",
        "exercise_tip3_desc": "• Da paseos cortos durante el día para interrumpir largos períodos de estar sentado. ¡Los pequeños movimientos suman!",
        "stress_tips": "Manejo del Estrés",
        "stress_tip1_title": "Atención Plena",
        "stress_tip1_desc": "• Practica **meditación, yoga o mindfulness** para reducir el estrés. Estas actividades pueden disminuir tu frecuencia cardíaca y presión arterial.",
        "stress_tip2_title": "Sueño de Calidad",
        "stress_tip2_desc": "• Duerme lo suficiente (**7-8 horas por noche**) para el bienestar general. La falta de sueño está relacionada con un mayor riesgo de enfermedad cardíaca.",
        "stress_tip3_title": "Conecta con Otros",
        "stress_tip3_desc": "• Dedica tiempo a pasatiempos y a tus seres queridos para relajarte y encontrar alegría. Una red social fuerte es excelente para la salud del corazón.",
        "settings_title": "Configuración",
        "app_customization": "Personalización de la aplicación",
        "theme": "Tema",
        "language": "Idioma",
        "lang_info": "Idioma cambiado a {lang}. (Nota: el contenido del texto no está completamente traducido para esta demostración).",
        "navigation": "Navegación",
        "dashboard": "Panel",
        "predict": "Predecir",
        "reports": "Informes",
        "tips": "Consejos",
        "settings": "Configuración",
        "logout": "Cerrar Sesión",
        "login_prompt": "Por favor, inicia sesión para navegar por la aplicación.",
        "app_footer": "Hecho por Amna",
        "font_settings": "Configuración de la Fuente",
        "font_size": "Tamaño de la Fuente del Cuerpo",
        "data_management": "Gestión de Datos",
        "clear_history": "Borrar historial al cerrar sesión",
        "reset_app": "Reiniciar la Aplicación",
        "reset_info": "Esto borrará todos los datos y reiniciará la aplicación.",
        "patient_name_label": "Nombre del Paciente:"
    }
}

THEMES = {
    "Dark": {
        "primary": "#DC2626",
        "background": "#121212",
        "secondary_bg": "#1F1F1F",
        "text_color": "#FFFFFF"
    },
    "Light": {
        "primary": "#DC2626",
        "background": "#FFFFFF",
        "secondary_bg": "#F0F2F6",
        "text_color": "#000000"
    },
    "Blue": {
        "primary": "#3B82F6",
        "background": "#095CAA",
        "secondary_bg": "#324C80",
        "text_color": "#DCE5F1"
    },
    "Green": {
        "primary": "#10B981",
        "background": "#3AC25E",
        "secondary_bg": "#D1FAE5",
        "text_color": "#1F2937"
    }
}

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'Dark'
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'font_size' not in st.session_state:
    st.session_state.font_size = 14
if 'clear_history_on_logout' not in st.session_state:
    st.session_state.clear_history_on_logout = False

# --- Page Navigation Functions ---
def set_page(page_name):
    """Function to change the current page in session state."""
    st.session_state.page = page_name

def add_footer():
    """Adds a footer to the page."""
    lang = LANGUAGES[st.session_state.language]
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: gray;'>{lang['app_footer']}</p>", unsafe_allow_html=True)


def reset_app():
    """Resets all session state variables."""
    st.session_state.page = 'welcome'
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.prediction_result = None
    st.session_state.prediction_history = []
    st.session_state.font_size = 14
    st.session_state.clear_history_on_logout = False
    st.session_state.theme_mode = 'Dark'
    st.session_state.language = 'English'


def welcome_page():
    """Renders the welcome page."""
    lang = LANGUAGES[st.session_state.language]
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(f"<h1 style='text-align: center; color: {THEMES[st.session_state.theme_mode]['primary']}; font-size: 3rem;'>{lang['title']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.25rem; color: {THEMES[st.session_state.theme_mode]['text_color']};'>{lang['subtitle']}</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(lang['get_started'], use_container_width=True):
            set_page('login')
    add_footer()

def login_page():
    """Renders the login/signup page."""
    lang = LANGUAGES[st.session_state.language]
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(f"<h2 style='text-align: center;'>{lang['login_title']}</h2>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input(lang['name'])
            email = st.text_input(lang['email'])
            password = st.text_input(lang['password'], type="password")
            submitted = st.form_submit_button(lang['proceed'])
            
            if submitted:
                if username and email and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = 'dashboard'
                    st.success(lang['login_success'])
                    st.rerun()
                else:
                    st.error(lang['login_error'])
    add_footer()

def dashboard_page():
    """Renders the user dashboard."""
    lang = LANGUAGES[st.session_state.language]
    st.title(lang['dashboard_welcome'].format(username=st.session_state.username))
    st.markdown(f"<p>{lang['dashboard_subtitle']}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dashboard Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown(f"### {lang['check_risk']}")
            st.markdown(lang['check_risk_desc'])
            st.button(lang['predict_button'], on_click=lambda: set_page('predict'), use_container_width=True)
    with col2:
        with st.container(border=True):
            st.markdown(f"### {lang['view_reports']}")
            st.markdown(lang['view_reports_desc'])
            st.button(lang['reports'], on_click=lambda: set_page('reports'), use_container_width=True)
    with col3:
        with st.container(border=True):
            st.markdown(f"### {lang['health_tips']}")
            st.markdown(lang['health_tips_desc'])
            st.button(lang['health_tips'], on_click=lambda: set_page('tips'), use_container_width=True)
    add_footer()

def prediction_page():
    """Renders the prediction form and results."""
    lang = LANGUAGES[st.session_state.language]
    st.title(lang['predict_title'])
    st.markdown(f"<p>{lang['predict_subtitle']}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load the model
    model = load_model()

    def predict_risk(data):
        """Makes a prediction using the loaded ML model."""
        if not model:
            return None, None
        # Convert data dictionary to DataFrame
        input_df = pd.DataFrame([data], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        # Make a prediction
        prediction = model.predict(input_df)[0]
        # Get the confidence score (probability of the predicted class)
        probabilities = model.predict_proba(input_df)[0]
        confidence = probabilities[prediction]
        
        is_high_risk = bool(prediction == 1) # Assuming 1 is the high-risk class
        return is_high_risk, confidence

    # --- Form Input Sections ---
    with st.expander(lang['personal_info'], expanded=True):
        age_col, sex_col = st.columns(2)
        with age_col:
            age = st.number_input(lang['age'], min_value=1, max_value=120, value=30)
        with sex_col:
            sex_options_display = [lang['male'], lang['female']]
            sex = st.selectbox(lang['sex'], options=sex_options_display)
            sex_encoded = 1 if sex == lang['male'] else 0

    with st.expander(lang['clinical_data'], expanded=True):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            cp = st.selectbox(lang['cp'], options=lang['cp_options'])
            trestbps = st.number_input(lang['trestbps'], min_value=50, max_value=250, value=120)
            chol = st.number_input(lang['chol'], min_value=100, max_value=600, value=200)
            fbs = st.radio(lang['fbs'], options=lang['fbs_options'])

        with col_c2:
            restecg = st.selectbox(lang['restecg'], options=lang['restecg_options'])
            thalach = st.number_input(lang['thalach'], min_value=60, max_value=220, value=150)
            exang = st.radio(lang['exang'], options=lang['exang_options'])
            oldpeak = st.number_input(lang['oldpeak'], min_value=0.0, max_value=6.2, value=1.0)
            slope = st.selectbox(lang['slope'], options=lang['slope_options'])
            ca = st.selectbox(lang['ca'], options=['0', '1', '2', '3'])
            thal = st.selectbox(lang['thal'], options=lang['thal_options'])

    if st.button(lang['predict_button'], use_container_width=True):
        user_data = {
            'age': age,
            'sex': sex_encoded,
            'cp': int(cp[-2]),
            'trestbps': trestbps,
            'chol': chol,
            'fbs': int(fbs[-2]),
            'restecg': int(restecg[-2]),
            'thalach': thalach,
            'exang': int(exang[-2]),
            'oldpeak': oldpeak,
            'slope': int(slope[-2]),
            'ca': int(ca),
            'thal': int(thal[-2])
        }
        
        with st.spinner(lang['predicting']):
            is_high_risk, confidence = predict_risk(user_data)
        
        prediction_record = {
            'username': st.session_state.username,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_high_risk': is_high_risk,
            'confidence': confidence,
            'data': user_data
        }
        st.session_state.prediction_result = prediction_record
        st.session_state.prediction_history.append(prediction_record)

    # --- Prediction Result Display ---
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        st.subheader(lang['result_title'])
        
        if result['is_high_risk']:
            st.markdown(f"""
            <div style="background-color: #F87171; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                <h3><b>{lang['high_risk']}</b></h3>
                <p>{lang['high_risk_msg']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #34D399; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                <h3><b>{lang['low_risk']}</b></h3>
                <p>{lang['low_risk_msg']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Score Chart
        fig = go.Figure(data=[go.Pie(
            labels=[lang['confidence'], ''],
            values=[result['confidence'], 1 - result['confidence']],
            marker_colors=[THEMES[st.session_state.theme_mode]['primary'], '#D1D5DB'],
            hole=0.5
        )])
        fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<p style='text-align: center;'><b>{lang['confidence']}</b> {result['confidence']*100:.2f}%</p>", unsafe_allow_html=True)

        # Download Report Button
        report_text = f"""
{lang['report_header']}
------------------------------------
{lang['patient_name_label']} {result['username']}
{lang['report_pred']} {'High Risk' if result['is_high_risk'] else 'Low Risk'}
{lang['report_conf']} {result['confidence']:.2f}

Patient Data:
Age: {result['data']['age']}
Sex: {'Male' if result['data']['sex'] == 1 else 'Female'}
Chest Pain Type: {result['data']['cp']}
Resting Blood Pressure: {result['data']['trestbps']}
Cholesterol: {result['data']['chol']}
Fasting Blood Sugar > 120 mg/dl: {result['data']['fbs']}
Resting ECG Results: {result['data']['restecg']}
Maximum Heart Rate Achieved: {result['data']['thalach']}
Exercise-induced Angina: {result['data']['exang']}
ST Depression: {result['data']['oldpeak']}
Slope of ST Segment: {result['data']['slope']}
Number of Major Vessels: {result['data']['ca']}
Thalassemia: {result['data']['thal']}
------------------------------------
"""
        st.download_button(
            label=lang['download_report'],
            data=report_text,
            file_name="heart_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    add_footer()

def reports_page():
    """Renders the reports page."""
    lang = LANGUAGES[st.session_state.language]
    st.title(lang['reports_title'])
    
    if not st.session_state.prediction_history:
        st.markdown(f"<p>{lang['reports_empty']}</p>", unsafe_allow_html=True)
    else:
        st.markdown("---")
        for i, report in enumerate(reversed(st.session_state.prediction_history)):
            report_status = lang['high_risk'] if report['is_high_risk'] else lang['low_risk']
            color = THEMES[st.session_state.theme_mode]['primary'] if report['is_high_risk'] else "#34D399"
            
            with st.expander(f"**Report {len(st.session_state.prediction_history) - i}** - {report['timestamp']} - **{report_status}**", expanded=False):
                st.write(f"**{lang['patient_name_label']}** {report['username']}")
                st.write(f"**{lang['report_pred']}**: <span style='color:{color}'>{report_status}</span>", unsafe_allow_html=True)
                st.write(f"**{lang['report_conf']}** {report['confidence']*100:.2f}%")
                st.markdown("---")
                st.subheader("Patient Data")
                # Display data in a table or list format
                report_data_list = [
                    (lang['age'], report['data']['age']),
                    (lang['sex'], lang['male'] if report['data']['sex'] == 1 else lang['female']),
                    (lang['cp'], report['data']['cp']),
                    (lang['trestbps'], report['data']['trestbps']),
                    (lang['chol'], report['data']['chol']),
                    (lang['fbs'], report['data']['fbs']),
                    (lang['restecg'], report['data']['restecg']),
                    (lang['thalach'], report['data']['thalach']),
                    (lang['exang'], report['data']['exang']),
                    (lang['oldpeak'], report['data']['oldpeak']),
                    (lang['slope'], report['data']['slope']),
                    (lang['ca'], report['data']['ca']),
                    (lang['thal'], report['data']['thal'])
                ]
                st.table(pd.DataFrame(report_data_list, columns=["Feature", "Value"]))
    add_footer()

def tips_page():
    """Renders the health tips page with enhanced layout."""
    lang = LANGUAGES[st.session_state.language]
    st.title(lang['tips_title'])
    st.markdown(f"<p>{lang['tips_subtitle']}</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Layout using columns for better organization
    diet_col, exercise_col, stress_col = st.columns(3)

    with diet_col:
        st.markdown(f"### {lang['diet_tips']}")
        st.markdown(f"**{lang['diet_tip1_title']}**")
        st.markdown(lang['diet_tip1_desc'])
        st.markdown(f"**{lang['diet_tip2_title']}**")
        st.markdown(lang['diet_tip2_desc'])
        st.markdown(f"**{lang['diet_tip3_title']}**")
        st.markdown(lang['diet_tip3_desc'])

    with exercise_col:
        st.markdown(f"### {lang['exercise_tips']}")
        st.markdown(f"**{lang['exercise_tip1_title']}**")
        st.markdown(lang['exercise_tip1_desc'])
        st.markdown(f"**{lang['exercise_tip2_title']}**")
        st.markdown(lang['exercise_tip2_desc'])
        st.markdown(f"**{lang['exercise_tip3_title']}**")
        st.markdown(lang['exercise_tip3_desc'])
    
    with stress_col:
        st.markdown(f"### {lang['stress_tips']}")
        st.markdown(f"**{lang['stress_tip1_title']}**")
        st.markdown(lang['stress_tip1_desc'])
        st.markdown(f"**{lang['stress_tip2_title']}**")
        st.markdown(lang['stress_tip2_desc'])
        st.markdown(f"**{lang['stress_tip3_title']}**")
        st.markdown(lang['stress_tip3_desc'])
    add_footer()

def settings_page():
    """Renders the settings page with theme and language options."""
    lang = LANGUAGES[st.session_state.language]
    st.title(lang['settings_title'])
    st.markdown("---")
    
    st.subheader(lang['app_customization'])
    
    # Theme settings
    theme_options = list(THEMES.keys())
    new_theme_mode = st.selectbox(lang['theme'], options=theme_options, index=theme_options.index(st.session_state.theme_mode))
    if new_theme_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = new_theme_mode
        st.rerun()

    # Language settings
    language_options = list(LANGUAGES.keys())
    new_language = st.selectbox(lang['language'], options=language_options, index=language_options.index(st.session_state.language))
    if new_language != st.session_state.language:
        st.session_state.language = new_language
        st.info(lang['lang_info'].format(lang=new_language))
        st.rerun()

    st.subheader(lang['font_settings'])
    st.session_state.font_size = st.slider(lang['font_size'], min_value=12, max_value=20, value=st.session_state.font_size)
    st.markdown("---")

    st.subheader(lang['data_management'])
    st.session_state.clear_history_on_logout = st.checkbox(lang['clear_history'], value=st.session_state.clear_history_on_logout)
    
    if st.button(lang['reset_app']):
        st.warning(lang['reset_info'])
        st.button("Confirm Reset", on_click=reset_app)
    add_footer()

# --- Main App Logic ---
# Apply dynamic theme and font size based on session state
current_theme = THEMES[st.session_state.theme_mode]
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {current_theme['background']};
        color: {current_theme['text_color']};
        font-size: {st.session_state.font_size}px;
    }}
    .stRadio, .stSelectbox, .stNumberInput, .st-bh, .st-bl, .st-bm, .st-bn, .st-cg, .st-b, .st-e, .st-cs {{
        color: {current_theme['text_color']} !important;
    }}
    .st-bv {{
        color: {current_theme['text_color']} !important;
        background-color: {current_theme['secondary_bg']};
    }}
    .report-table th, .report-table td {{
        background-color: {current_theme['secondary_bg']};
        color: {current_theme['text_color']};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(f"<h1 style='color: {current_theme['primary']}'>{LANGUAGES[st.session_state.language]['navigation']}</h1>", unsafe_allow_html=True)
# Sidebar navigation buttons (always visible)
if st.session_state.logged_in:
    st.sidebar.markdown(f"<h3 style='color: {current_theme['primary']}'>{LANGUAGES[st.session_state.language]['dashboard_welcome'].format(username=st.session_state.username)}</h3>", unsafe_allow_html=True)
    st.sidebar.button(LANGUAGES[st.session_state.language]['dashboard'], use_container_width=True, on_click=lambda: set_page('dashboard'))
    st.sidebar.button(LANGUAGES[st.session_state.language]['predict'], use_container_width=True, on_click=lambda: set_page('predict'))
    st.sidebar.button(LANGUAGES[st.session_state.language]['reports'], use_container_width=True, on_click=lambda: set_page('reports'))
    st.sidebar.button(LANGUAGES[st.session_state.language]['tips'], use_container_width=True, on_click=lambda: set_page('tips'))
    st.sidebar.button(LANGUAGES[st.session_state.language]['settings'], use_container_width=True, on_click=lambda: set_page('settings'))
    st.sidebar.markdown("---")
    if st.sidebar.button(LANGUAGES[st.session_state.language]['logout'], use_container_width=True):
        if st.session_state.clear_history_on_logout:
            st.session_state.prediction_history = []
        st.session_state.logged_in = False
        st.session_state.username = ''
        set_page('welcome')
else:
    # Buttons for unauthenticated users
    st.sidebar.button(LANGUAGES[st.session_state.language]['get_started'], use_container_width=True, on_click=lambda: set_page('login'))
    st.sidebar.markdown(f"<p>{LANGUAGES[st.session_state.language]['login_prompt']}</p>", unsafe_allow_html=True)
    st.sidebar.button(LANGUAGES[st.session_state.language]['tips'], use_container_width=True, on_click=lambda: set_page('tips'))
    st.sidebar.button(LANGUAGES[st.session_state.language]['settings'], use_container_width=True, on_click=lambda: set_page('settings'))

# Conditional page rendering based on session state
if st.session_state.logged_in:
    if st.session_state.page == 'dashboard':
        dashboard_page()
    elif st.session_state.page == 'predict':
        prediction_page()
    elif st.session_state.page == 'reports':
        reports_page()
    elif st.session_state.page == 'tips':
        tips_page()
    elif st.session_state.page == 'settings':
        settings_page()
else:
    if st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'tips':
        tips_page()
    elif st.session_state.page == 'settings':
        settings_page()
    else:
        welcome_page()
