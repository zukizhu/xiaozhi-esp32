#ifndef __HIMAX6538_H__
#define __HIMAX6538_H__

#include "driver/uart.h"
#include "driver/gpio.h"

#define UART_PORT UART_NUM_1
#define BUF_SIZE (1024)

class Himax6538 {
public:
    Himax6538();
    ~Himax6538();   
    static void Get_Uart_data(char* json_str);
    static void Setup_Himax_task();

private:
    static void Uart_task(void *pvParameters);
};

#endif
