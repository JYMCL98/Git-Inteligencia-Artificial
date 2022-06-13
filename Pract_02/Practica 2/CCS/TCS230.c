void TCS230_init(void)                           // Inicializa el sensor
{
   setup_timer_1(T1_INTERNAL | T1_DIV_BY_8);     // Inicializa el timer 1 con un preescaler de 8
}

void TSC230_clear(void)
{
   output_high(S2);
   output_low(S3);
   while(input_state(P_TCS230)){
   }
   set_timer1(0); 
   while(!input_state(P_TCS230)){
   }
}

unsigned long TCS230_getFrequence_red(void)      // Funcion para obtencion de valores del canal rojo
{
   TSC230_clear();
   unsigned long frequence_red = 0;
   output_low(S2);
   output_low(S3);
   while(input_state(P_TCS230)){
   } 
   set_timer1(0); 
   while(!input_state(P_TCS230)){
   } 
   frequence_red = get_timer1();
   return frequence_red;
}

unsigned long TCS230_getFrequence_green(void)    // Funcion para obtencion de valores del canal verde
{
   TSC230_clear();
   unsigned long frequence_green = 0;
   output_high(S2);
   output_high(S3);
   while(input_state(P_TCS230)){
   } 
   set_timer1(0); 
   while(!input_state(P_TCS230)){
   } 
   frequence_green = get_timer1();
   return frequence_green;
}

unsigned long TCS230_getFrequence_blue(void)     // Funcion para obtencion de valores del canal azul
{
   TSC230_clear();
   unsigned long frequence_blue = 0;
   output_low(S2);
   output_high(S3);
   while(input_state(P_TCS230)){
   } 
   set_timer1(0); 
   while(!input_state(P_TCS230)){
   } 
   frequence_blue = get_timer1();
   return frequence_blue;
}
