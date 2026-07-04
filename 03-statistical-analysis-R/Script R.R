# =====================================================================
# 1. CARGA DE DATOS Y PREPARACIÓN INICIAL
# =====================================================================
datos <- read.csv("Hotel reservations c.csv", sep = ";", dec = ",")

# Crear variable objetivo binaria (1 = Canceled, 0 = Not_Canceled)
datos$cancelacion <- ifelse(datos$booking_status == "Canceled", 1, 0)

# Eliminar columnas que no aportan valor predictivo o están duplicadas
datos$Booking_ID <- NULL
datos$booking_status <- NULL

# =====================================================================
# 2. LIMPIEZA Y CREACIÓN DE "datos_limpios" (NUESTRA BASE DEFINITIVA)
# =====================================================================
# Filtramos de golpe los dos segmentos que no nos interesan
datos_limpios <- subset(datos, market_segment_type != "Aviation" & market_segment_type != "Complementary")

# =====================================================================
# 3. TRANSFORMACIÓN DE VARIABLES (TRABAJAMOS SOLO CON datos_limpios)
# =====================================================================
# Convertimos las variables categóricas en factores
datos_limpios$room_type_reserved <- as.factor(datos_limpios$room_type_reserved)
datos_limpios$type_of_meal_plan <- as.factor(datos_limpios$type_of_meal_plan)
datos_limpios$market_segment_type <- as.factor(datos_limpios$market_segment_type)
datos_limpios$arrival_month <- as.factor(datos_limpios$arrival_month)

# =====================================================================
# 4. AJUSTE DE NIVELES Y REFERENCIAS 
# =====================================================================
# Borramos el recuerdo (niveles fantasma) de Aviation y Complementary
datos_limpios$market_segment_type <- droplevels(datos_limpios$market_segment_type)

# Forzamos a que "Corporate" sea el grupo de referencia (Base invisible)
datos_limpios$market_segment_type <- relevel(datos_limpios$market_segment_type, ref = "Corporate")

# Forzamos a que "Enero (1)" sea el mes de referencia (Base invisible)
datos_limpios$arrival_month <- relevel(datos_limpios$arrival_month, ref = "1")


# =====================================================================
# MODELO 1
# =====================================================================

modelo_glm_1 <- glm(cancelacion ~ market_segment_type + 
                      no_of_special_requests + 
                      avg_price_per_room + 
                      arrival_month + 
                      repeated_guest + 
                      lead_time +
                      required_car_parking_space +  
                      no_of_weekend_nights,         
                    data = datos_limpios, 
                    family = binomial(link = "logit"))

summary(modelo_glm_1)

# =====================================================================
# MODELO 2
# =====================================================================

modelo_glm_2<- glm(cancelacion ~ market_segment_type + 
                                no_of_special_requests + 
                                avg_price_per_room + 
                                arrival_month + 
                                repeated_guest + 
                                lead_time +
                                required_car_parking_space +  
                                no_of_weekend_nights +
                                lead_time:avg_price_per_room +               # Interacción 1 (Precio/Antelación)
                                lead_time:repeated_guest,                    # Interacción 3 (Antelación/Fidelidad)
                              family = binomial(link = "logit"), 
                              data = datos_limpios)

summary(modelo_glm_2)


# =====================================================================
# MODELO 3
# =====================================================================

modelo_glm_3 <- glm(cancelacion ~ market_segment_type + 
                      no_of_special_requests + 
                      avg_price_per_room + 
                      arrival_month + 
                      repeated_guest + 
                      lead_time +
                      required_car_parking_space +  
                      no_of_weekend_nights +
                      avg_price_per_room:lead_time +       
                      repeated_guest:lead_time +           
                      I(lead_time^2) +                     # Efecto Techo: Antelación
                      I(avg_price_per_room^2),             # Efecto Techo: Precio
                    family = binomial(link = "logit"), 
                    data = datos_limpios)

# 1. Ver el resumen
summary(modelo_glm_3)

anova(modelo_glm_3, test = "Chisq")

# 1. Obtener las probabilidades de cancelación que predice el modelo
probabilidades <- predict(modelo_glm_3, type = "response", newdata = datos_limpios)

# ==========================================
# 2. ESCENARIO BASE: CORTE AL 50%
# ==========================================
predicciones_50 <- ifelse(probabilidades > 0.5, 1, 0)
matriz_50 <- table(Real = datos_limpios$cancelacion, Prediccion = predicciones_50)

# Fórmula de la fiabilidad (Aciertos / Total)
fiabilidad_50 <- sum(diag(matriz_50)) / sum(matriz_50)

cat("\n--- RESULTADOS AL 50% ---\n")
print(matriz_50)
cat("Fiabilidad Global (Accuracy) al 50%: ", round(fiabilidad_50 * 100, 2), "%\n")


# ==========================================
# 3. ESCENARIO ESTRATÉGICO: CORTE AL 70%
# ==========================================
predicciones_70 <- ifelse(probabilidades > 0.7, 1, 0)
matriz_70 <- table(Real = datos_limpios$cancelacion, Prediccion = predicciones_70)

# Fórmula de la fiabilidad (Aciertos / Total)
fiabilidad_70 <- sum(diag(matriz_70)) / sum(matriz_70)

cat("\n--- RESULTADOS AL 70% ---\n")
print(matriz_70)
cat("Fiabilidad Global (Accuracy) al 70%: ", round(fiabilidad_70 * 100, 2), "%\n")


