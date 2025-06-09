#include "himax6538.h"
#include "board.h"
#include "display.h"
#include "esp_io_expander.h"
#include "cJSON.h"

#include <esp_log.h>

#define TAG "Himax6538"

Himax6538::Himax6538() {
    // Initialize UART for Himax6538 communication
    // UART 配置（波特率 115200，8N1）
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_param_config(UART_PORT, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(UART_PORT, GPIO_NUM_17, GPIO_NUM_18, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, BUF_SIZE * 2, 0, 0, NULL, 0));

    // 摄像头控制引脚初始化
    gpio_reset_pin(GPIO_NUM_33); // PWDN
    gpio_set_direction(GPIO_NUM_33, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_NUM_34); // RESET
    gpio_set_direction(GPIO_NUM_34, GPIO_MODE_OUTPUT);

    // 唤醒摄像头
    gpio_set_level(GPIO_NUM_33, 0); // 退出休眠
    gpio_set_level(GPIO_NUM_34, 0); // 复位脉冲（>1ms）
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(GPIO_NUM_34, 1);
    ESP_LOGI(TAG, "Himax6538 UART 驱动已安装");
}

Himax6538::~Himax6538() {
    // 卸载 UART 驱动
    uart_driver_delete(UART_PORT);
    ESP_LOGI(TAG, "Himax6538 UART 驱动已卸载");
}

void Himax6538::Get_Uart_data(char* json_str) {

    cJSON *root = cJSON_Parse(json_str);
    if (root == NULL) {
        ESP_LOGE(TAG, "Uart JSON 解析失败");
        return;
    }

    cJSON *objects = cJSON_GetObjectItem(root, "objects");
    if (objects != NULL) {
        int obj_count = cJSON_GetArraySize(objects);
        for (int i = 0; i < obj_count; i++) {
            cJSON *obj = cJSON_GetArrayItem(objects, i);
            cJSON *type = cJSON_GetObjectItem(obj, "type");
            cJSON *conf = cJSON_GetObjectItem(obj, "confidence");
            ESP_LOGI(TAG, "目标 %d: %s (置信度: %.2f)", i+1, type->valuestring, conf->valuedouble);
        }
    }
    cJSON_Delete(root);
    ESP_LOGI(TAG, "Uart 数据获取成功");
}

void Himax6538::Uart_task(void *pvParameters) {

    uint8_t data[BUF_SIZE];

    while (1) {
        int len = uart_read_bytes(UART_PORT, data, BUF_SIZE, pdMS_TO_TICKS(100));
        if (len > 0) {
            data[len] = '\0'; // 添加字符串终止符
            ESP_LOGI(TAG, "Uart接收: %s", data);
            Get_Uart_data((char*)data);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void Himax6538::Setup_Himax_task() {
    xTaskCreate(Himax6538::Uart_task, "Himax6538_Uart_Task", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "Himax6538 Uart Task 已创建");
}
