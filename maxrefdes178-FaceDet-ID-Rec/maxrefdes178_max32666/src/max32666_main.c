/*******************************************************************************
 * Copyright (C) 2020-2021 Maxim Integrated Products, Inc., All rights Reserved.
 *
 * This software is protected by copyright laws of the United States and
 * of foreign countries. This material may also be protected by patent laws
 * and technology transfer regulations of the United States and of foreign
 * countries. This software is furnished under a license agreement and/or a
 * nondisclosure agreement and may only be used or reproduced in accordance
 * with the terms of those agreements. Dissemination of this information to
 * any party or parties not specified in the license agreement and/or
 * nondisclosure agreement is expressly prohibited.
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of Maxim Integrated
 * Products, Inc. shall not be used except as stated in the Maxim Integrated
 * Products, Inc. Branding Policy.
 *
 * The mere transfer of this software does not imply any licenses
 * of trade secrets, proprietary technology, copyrights, patents,
 * trademarks, maskwork rights, or any other form of intellectual
 * property whatsoever. Maxim Integrated Products, Inc. retains all
 * ownership rights.
 *******************************************************************************
 */

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#include <board.h>
#include <core1.h>
#include <dma.h>
#include <icc_regs.h>
#include <mxc_delay.h>
#include <mxc_sys.h>
#include <nvic_table.h>
#include <stdint.h>
#include <string.h>
#include <tmr.h>
#include <stdlib.h>

#include "max32666_accel.h"
#include "max32666_audio_codec.h"
//#include "max32666_ble.h"
//#include "max32666_ble_command.h"
//#include "max32666_ble_queue.h"
#include "max32666_data.h"
#include "max32666_debug.h"
#include "max32666_expander.h"
#include "max32666_ext_flash.h"
#include "max32666_ext_sram.h"
#include "max32666_fonts.h"
#include "max32666_fuel_gauge.h"
#include "max32666_i2c.h"
#include "max32666_lcd.h"
#include "max32666_lcd_images.h"
#include "max32666_pmic.h"
#include "max32666_powmon.h"
#include "max32666_qspi_master.h"
#include "max32666_sdcard.h"
#include "max32666_spi_dma.h"
#include "max32666_timer_led_button.h"
#include "max32666_touch.h"
#include "max32666_usb.h"
#include "maxrefdes178_definitions.h"
#include "maxrefdes178_version.h"
#include "max32666_record.h"
#include "max32666_embeddings.h"

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#define S_MODULE_NAME   "main"

#define ML_DATA_SIZE    5
#define BOX_THICKNESS   3
//-----------------------------------------------------------------------------
// Typedefs
//-----------------------------------------------------------------------------
typedef struct {
    char*    data;
    int len;
} text_t;
//-----------------------------------------------------------------------------
// Global variables
//-----------------------------------------------------------------------------
char names[1024][7]; // 1024 names of 6 bytes each, as we support 1024 people in the database
extern int record;
static volatile int core1_init_done = 0;
static char lcd_string_buff[LCD_NOTIFICATION_MAX_SIZE] = {0};
static char version_string[14] = {0};
static char usn_string[(sizeof(serial_num_t) + 1) * 3] = {0};
static char mac_string[(sizeof(device_info.ble_mac) + 1) * 3] = {0};
static uint16_t video_string_color;
static uint16_t video_frame_color;
static uint16_t audio_string_color;
int face_detected = 0;
int getting_name = 1;
int capture = 0;
int record_flag = 0;
int key = 0;
int block_button_x = 0;
static text_t label_text[] = {
    // info
    {(char*)"Face", 4},
};
//-----------------------------------------------------------------------------
// Local function declarations
//-----------------------------------------------------------------------------
static void core0_irq_init(void);
static void core1_irq_init(void);
static void core0_icc(int enable);
static void core1_icc(int enable);
static void run_application(void);
static int refresh_screen(void);
static void digit_detection_result(void);
//-----------------------------------------------------------------------------
// Function definitions
//-----------------------------------------------------------------------------
int main(void)
{
    int ret = 0;

    // Set PORT1 and PORT2 rail to VDDIO
    MXC_GPIO0->vssel = 0x00;
    MXC_GPIO1->vssel = 0x00;

    // Disable Core0 ICC0 Instruction cache
    core0_icc(0);

    ret = MXC_SEMA_Init();
    if (ret != E_NO_ERROR) {
        printf("MXC_SEMA_Init failed %d\n", ret);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }

    device_info.device_version.max32666.major = S_VERSION_MAJOR;
    device_info.device_version.max32666.minor = S_VERSION_MINOR;
    device_info.device_version.max32666.build = S_VERSION_BUILD;
    snprintf(version_string, sizeof(version_string) - 1, "v%d.%d.%d", S_VERSION_MAJOR, S_VERSION_MINOR, S_VERSION_BUILD);
    snprintf(device_info.max32666_demo_name, sizeof(device_info.max32666_demo_name) - 1, "%s", FACIAL_RECOGNITION_DEMO_NAME);
    PR("\n\n\n");
    PR_INFO("maxrefdes178_max32666 %s core0 %s [%s]", device_info.max32666_demo_name, version_string, S_BUILD_TIMESTAMP);

    ret = i2c_master_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("i2c_init failed %d", ret);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }

    ret = expander_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("expander_init failed %d", ret);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }

    ret = pmic_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("pmic_init failed %d", ret);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }
    pmic_led_green(1);

   // BLE should init first since it is mischievous
   // BLE init somehow damages GPIO settings for P0.0, P0.23
    core0_irq_init();
    Core1_Start();

    for(uint32_t cnt = 10000000; !core1_init_done && cnt; cnt--) {
        if (cnt == 1) {
            PR_ERROR("timeout, reset");
            MXC_Delay(MXC_DELAY_MSEC(100));
            pmic_hard_reset();
            MXC_Delay(MXC_DELAY_MSEC(100));
            MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
        }
    }

    ret = fuel_gauge_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("fuel_gauge_init failed %d", ret);
        pmic_led_red(1);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }

    // Initialize DMA peripheral
    ret = MXC_DMA_Init(MXC_DMA0);
    if (ret != E_NO_ERROR) {
        PR_ERROR("MXC_DMA_Init failed %d", ret);
        pmic_led_red(1);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }


    ret = lcd_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("lcd_init failed %d", ret);
        pmic_led_red(1);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }

    ret = qspi_master_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("qspi_master_init failed %d", ret);
        pmic_led_red(1);
        MXC_Delay(MXC_DELAY_MSEC(100));
        MXC_SYS_Reset_Periph(MXC_SYS_RESET_SYSTEM);
    }

    ret = timer_led_button_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("timer_led_button_init failed %d", ret);
        pmic_led_red(1);
    }

    ret = accel_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("accel_init failed %d", ret);
        pmic_led_red(1);
    }

    ret = touch_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("touch_init failed %d", ret);
        pmic_led_red(1);
    }


    ret = sdcard_init();
    if (ret != E_NO_ERROR) {
        PR_ERROR("sdcard_init failed %d", ret);
        pmic_led_red(1);
        fonts_putString(1, 14, "SD card not found.    This example requires SD Card.", &Font_11x18, RED, 1, BLACK, lcd_data.buffer);
        lcd_drawImage(lcd_data.buffer);
        MXC_Delay(MXC_DELAY_MSEC(1000));
        while(1);
    }
	uint8_t checksum[MXC_SYS_USN_CHECKSUM_LEN];
    ret = MXC_SYS_GetUSN(device_info.device_serial_num.max32666, checksum);

    if (ret != E_NO_ERROR) {
        PR_ERROR("MXC_SYS_GetUSN failed %d", ret);
        pmic_led_red(1);
    }
    snprintf(usn_string, sizeof(usn_string) - 1, "%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X%02X",
            device_info.device_serial_num.max32666[0],
            device_info.device_serial_num.max32666[1],
            device_info.device_serial_num.max32666[2],
            device_info.device_serial_num.max32666[3],
            device_info.device_serial_num.max32666[4],
            device_info.device_serial_num.max32666[5],
            device_info.device_serial_num.max32666[6],
            device_info.device_serial_num.max32666[7],
            device_info.device_serial_num.max32666[8],
            device_info.device_serial_num.max32666[9],
            device_info.device_serial_num.max32666[10],
            device_info.device_serial_num.max32666[11],
            device_info.device_serial_num.max32666[12]);
    PR_INFO("MAX32666 Serial number: %s", usn_string);


    /* Select USB-Type-C Debug Connection to MAX78000-Video on IO expander */
    if ((ret = expander_select_debugger(DEBUGGER_SELECT_MAX78000_VIDEO)) != E_NO_ERROR) {
       PR_ERROR("expander_debug_select failed %d", ret);
       pmic_led_red(1);
    }

    // Print logo and version
    fonts_putStringCentered(LCD_HEIGHT - 66, version_string, &Font_16x26, GRED, adi_logo);
    fonts_putStringCentered(LCD_HEIGHT - 38, mac_string, &Font_11x18, BLUE, adi_logo);
    fonts_putStringCentered(3, usn_string, &Font_7x10, LGRAY, adi_logo); //change to light grey to match the new background
    fonts_putStringCentered(55, device_info.max32666_demo_name, &Font_16x26, MAGENTA, adi_logo);
    lcd_drawImage(adi_logo);

    // Wait MAX78000s
    MXC_Delay(MXC_DELAY_MSEC(3000));

    // Get information from MAX78000
    {
        qspi_packet_type_e qspi_packet_type_rx = 0;
        
        for (int try = 0; try < 3; try++) {
            qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_FACEID_SUBJECTS_CMD);
            qspi_master_wait_video_int();
            qspi_master_video_rx_worker(&qspi_packet_type_rx);
            if (device_status.faceid_embed_subject_names_size) {
                break;
            }
        }

        for (int try = 0; try < 3; try++) {
            qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_VERSION_CMD);
            qspi_master_wait_video_int();
            qspi_master_video_rx_worker(&qspi_packet_type_rx);
            if (device_info.device_version.max78000_video.major || device_info.device_version.max78000_video.minor) {
                break;
            }
        }
        for (int try = 0; try < 3; try++) {
            qspi_master_send_audio(NULL, 0, QSPI_PACKET_TYPE_AUDIO_VERSION_CMD);
            qspi_master_wait_audio_int();
            qspi_master_audio_rx_worker(&qspi_packet_type_rx);
            if (device_info.device_version.max78000_audio.major || device_info.device_version.max78000_audio.minor) {
                break;
            }
        }

        for (int try = 0; try < 3; try++) {
            qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_DEMO_NAME_CMD);
            qspi_master_wait_video_int();
            qspi_master_video_rx_worker(&qspi_packet_type_rx);
            if (device_info.max78000_video_demo_name[0]) {
                break;
            }
        }
        for (int try = 0; try < 3; try++) {
            qspi_master_send_audio(NULL, 0, QSPI_PACKET_TYPE_AUDIO_DEMO_NAME_CMD);
            qspi_master_wait_audio_int();
            qspi_master_audio_rx_worker(&qspi_packet_type_rx);
            if (device_info.max78000_audio_demo_name[0]) {
                break;
            }
        }

        ret = E_NO_ERROR;
        // Check video and audio fw version
        if (!(device_info.device_version.max78000_video.major || device_info.device_version.max78000_video.minor)) {
            PR_ERROR("max78000_video communication error");
            ret = E_COMM_ERR;
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "No video comm");
            fonts_putStringCentered(100, lcd_string_buff, &Font_11x18, RED, adi_logo);
        } else if (memcmp(&device_info.device_version.max32666, &device_info.device_version.max78000_video, sizeof(version_t))) {
            PR_ERROR("max32666 and max78000_video versions are different");
            ret = E_INVALID;
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "video fw err %d.%d.%d",
                    device_info.device_version.max78000_video.major,
                    device_info.device_version.max78000_video.minor,
                    device_info.device_version.max78000_video.build);
            fonts_putStringCentered(100, lcd_string_buff, &Font_11x18, RED, adi_logo);
        } else if (strncmp(device_info.max32666_demo_name, device_info.max78000_video_demo_name, DEMO_STRING_SIZE)) {
            PR_ERROR("max32666 and max78000_video demos are different");
            ret = E_INVALID;
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "video fw demo err %s", device_info.max78000_video_demo_name);
            fonts_putStringCentered(100, lcd_string_buff, &Font_11x18, RED, adi_logo);
        }

        if (!(device_info.device_version.max78000_audio.major || device_info.device_version.max78000_audio.minor)) {
            PR_ERROR("max78000_audio communication error");
            ret = E_COMM_ERR;
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "No audio comm");
            fonts_putStringCentered(130, lcd_string_buff, &Font_11x18, RED, adi_logo);
        } else if (memcmp(&device_info.device_version.max32666, &device_info.device_version.max78000_audio, sizeof(version_t))) {
            PR_ERROR("max32666 and max78000_audio versions are different");
            ret = E_INVALID;
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "audio fw err %d.%d.%d",
                    device_info.device_version.max78000_audio.major,
                    device_info.device_version.max78000_audio.minor,
                    device_info.device_version.max78000_audio.build);
            fonts_putStringCentered(130, lcd_string_buff, &Font_11x18, RED, adi_logo);
        } else if (strncmp(device_info.max32666_demo_name, device_info.max78000_audio_demo_name, DEMO_STRING_SIZE)) {
            PR_ERROR("max32666 and max78000_audio demos are different");
            ret = E_INVALID;
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "audio fw demo err %s", device_info.max78000_audio_demo_name);
            fonts_putStringCentered(130, lcd_string_buff, &Font_11x18, RED, adi_logo);
        }

        if (ret != E_NO_ERROR) {
            lcd_drawImage(adi_logo);
            pmic_led_red(1);
            while(1) {

                expander_worker();

                if (lcd_data.refresh_screen && !spi_dma_busy_flag(MAX32666_LCD_DMA_CHANNEL)) {
                    memcpy(lcd_data.buffer, adi_logo, sizeof(lcd_data.buffer));
                    if (strlen(lcd_data.notification) < (LCD_WIDTH / Font_11x18.width)) {
                        fonts_putStringCentered(LCD_HEIGHT - Font_11x18.height - 3, lcd_data.notification, &Font_11x18, lcd_data.notification_color, lcd_data.buffer);
                    } else {
                        fonts_putStringCentered(LCD_HEIGHT - Font_7x10.height - 3, lcd_data.notification, &Font_7x10, lcd_data.notification_color, lcd_data.buffer);
                    }
                    lcd_drawImage(lcd_data.buffer);
                }
            }
        }
    }

    PR_INFO("core 0 init completed");

    // print application name
    snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "Audio enabled");
    fonts_putStringCentered(37, lcd_string_buff, &Font_11x18, RED, adi_logo);

    run_application();

    return E_NO_ERROR;
}

static void run_application(void)
{
    qspi_packet_type_e qspi_packet_type_rx = 0;
    video_frame_color = WHITE;
    uint16_t touch_x1, touch_y1;
    uint32_t loop_time = 0;
    uint32_t refresh_time = 0;
    core0_icc(1);
    // Main application loop
    int db_number=0;
    char default_names[DEFAULT_EMBS_NUM][7] = DEFAULT_NAMES;
    // The code expects default embeddings from the user. 
    //The "db_gen" tool generates "weights_3.bin" and "max32666_embeddings.h" automatically. 
    //"weights_3.bin" should be put into the SD Card.
    #pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wstringop-truncation"  
	for (int i = 0; i < DEFAULT_EMBS_NUM; i++){
		strncpy((char*)names[i], default_names[i], 7);
	}
    #pragma GCC diagnostic pop
    find_names_number(names,&db_number);
    embeddings.capture_number = 0;
    while (1) {
        loop_time = timer_ms_tick;
        // Handle Video QSPI RX
        if (qspi_master_video_rx_worker(&qspi_packet_type_rx) == E_NO_ERROR) {
            switch(qspi_packet_type_rx) {
            case QSPI_PACKET_TYPE_VIDEO_DATA_RES:
                timestamps.video_data_received = timer_ms_tick;
                lcd_data.refresh_screen = 1;
                break;

			case QSPI_PACKET_TYPE_FACEDET_VIDEO_DATA_RES:
                timestamps.video_data_received = timer_ms_tick;
                lcd_data.refresh_screen = 1;
                break;

			case QSPI_PACKET_TYPE_VIDEO_ML_RES: // show ML result
                digit_detection_result();
                timestamps.video_data_received = timer_ms_tick;
                lcd_data.refresh_screen = 1;
                break;
            
			case QSPI_PACKET_TYPE_VIDEO_EMBEDDING_VALUES:
            //"QSPI_PACKET_TYPE_VIDEO_EMBEDDING_VALUES" packet type is used to take the embeddings buffer from MAX78000_Video.
                embeddings.capture_number++;
                record_flag = 1;
                capture =0;
                break;
            case QSPI_PACKET_TYPE_VIDEO_CLASSIFICATION_RES:
                timestamps.activity_detected = timer_ms_tick;
                if (device_status.classification_video.classification == CLASSIFICATION_UNKNOWN) {
                    video_string_color = RED;
                    video_frame_color = RED;
                } else if (device_status.classification_video.classification == CLASSIFICATION_LOW_CONFIDENCE) {
                    video_string_color = YELLOW;
                    video_frame_color = YELLOW;
                } else if (device_status.classification_video.classification == CLASSIFICATION_DETECTED) {
                    video_string_color = GREEN;
                    video_frame_color = GREEN;
                } else if (device_status.classification_video.classification == CLASSIFICATION_NOTHING) {
                    video_frame_color = WHITE;
                }

                break;
            case QSPI_PACKET_TYPE_VIDEO_FACEID_EMBED_UPDATE_RES:
                qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_FACEID_SUBJECTS_CMD);
                lcd_notification(GREEN, "FaceID signature updated");
                break;
            case QSPI_PACKET_TYPE_VIDEO_FACEID_SUBJECTS_RES:
                timestamps.faceid_subject_names_received = timer_ms_tick;
                break;
            case QSPI_PACKET_TYPE_VIDEO_SENDING_EMBEDDINGS_FLAG:
                sending_embeddings();
                break;    
            default:
                break;
            }
        }

        // Handle Audio QSPI RX
        if (qspi_master_audio_rx_worker(&qspi_packet_type_rx) == E_NO_ERROR) {
            switch(qspi_packet_type_rx) {

            case QSPI_PACKET_TYPE_AUDIO_CLASSIFICATION_RES:
                timestamps.audio_result_received = timer_ms_tick;
                timestamps.activity_detected = timer_ms_tick;

                if (device_status.classification_audio.classification == CLASSIFICATION_UNKNOWN) {
                    audio_string_color = RED;
                } else if (device_status.classification_audio.classification == CLASSIFICATION_LOW_CONFIDENCE) {
                    audio_string_color = YELLOW;
                } else if (device_status.classification_audio.classification == CLASSIFICATION_DETECTED) {
                    audio_string_color = YELLOW;//GREEN;

                    if (strncmp(device_status.classification_audio.result, "OFF", 3) == 0) {
                    	PR_INFO("OFF");
                          device_settings.enable_lcd = 0;
                          lcd_backlight(0, 0);
                          qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_DISABLE_CMD);
                    } else if(strncmp(device_status.classification_audio.result, "ON", 2) == 0) {
                    	PR_INFO("ON");
                        device_settings.enable_lcd = 1;
                        if (device_settings.enable_max78000_video) {
                            qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_ENABLE_CMD);
                        }
                        lcd_backlight(1, MAX32666_LCD_BACKLIGHT_HIGH);

                        // also enable cnn
                        device_settings.enable_max78000_video_cnn = 1;
                        qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_ENABLE_CNN_CMD);

                    } else if (strncmp(device_status.classification_audio.result, "GO", 2) == 0) {
                    	PR_INFO("GO");
                        //Do nothing

                    } else if(strncmp(device_status.classification_audio.result, "STOP", 4) == 0) {
                    	PR_INFO("STOP");
                        //Do nothing


                    }
                }


                if (!device_settings.enable_max78000_video) {
                    lcd_data.refresh_screen = 1;
                }
                break;
            default:
                break;
            }
        }


        // Handle QSPI TX
        qspi_master_video_tx_worker();
        qspi_master_audio_tx_worker();
        if(record_flag){
            record_flag = 0;
            record_embeddings(embeddings.embeddings_name,embeddings.embeddings_buffer);
        }
        if (device_settings.enable_max78000_video) {
            //PASS
        } 
        else {
            // If video is disabled, draw logo and refresh periodically
            if ((timer_ms_tick - timestamps.screen_drew) > LCD_VIDEO_DISABLE_REFRESH_DURATION) {
                memcpy(lcd_data.buffer, adi_logo, sizeof(lcd_data.buffer));
                snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "Video disabled");
                fonts_putStringCentered(15, lcd_string_buff, &Font_11x18, RED, lcd_data.buffer);
                lcd_data.refresh_screen = 1;
            }
        }
    


        // Check inactivity
        if (device_settings.enable_inactivity) {
            if ((timer_ms_tick - timestamps.activity_detected) > INACTIVITY_LONG_DURATION) {
                if (device_status.inactivity_state != INACTIVITY_STATE_INACTIVE_LONG) {
                    device_status.inactivity_state = INACTIVITY_STATE_INACTIVE_LONG;
                    // Switch to inactive long state
                    lcd_backlight(0, 0);
                    qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_DISABLE_CMD);
                    PR_INFO("Inactive long");
                }
            } else if ((timer_ms_tick - timestamps.activity_detected) > INACTIVITY_SHORT_DURATION) {
                if (device_status.inactivity_state != INACTIVITY_STATE_INACTIVE_SHORT) {
                    device_status.inactivity_state = INACTIVITY_STATE_INACTIVE_SHORT;
                    // Switch to inactive short state
                    if (device_settings.enable_lcd) {
                        lcd_backlight(1, MAX32666_LCD_BACKLIGHT_LOW);
                    }
                    PR_INFO("Inactive short");
                }
            } else {
                if (device_status.inactivity_state != INACTIVITY_STATE_ACTIVE) {
                    device_status.inactivity_state = INACTIVITY_STATE_ACTIVE;
                    // Switch to active state
                    if (device_settings.enable_lcd) {
                        lcd_backlight(1, MAX32666_LCD_BACKLIGHT_HIGH);
                    }
                    if (device_settings.enable_max78000_video && device_settings.enable_lcd) {
                        MXC_Delay(MXC_DELAY_MSEC(600));
                        qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_ENABLE_CMD);
                    }
                    PR_INFO("Active");
                }
            }
        }

        // Check PMIC and Fuel Gauge
        if ((timer_ms_tick - timestamps.pmic_check) > MAX32666_PMIC_INTERVAL) {
            timestamps.pmic_check = timer_ms_tick;
            pmic_worker();
            if (device_status.fuel_gauge_working) {
                fuel_gauge_worker();
            }
        }


        // LED worker
        if ((timer_ms_tick - timestamps.led) > MAX32666_LED_INTERVAL) {
            timestamps.led = timer_ms_tick;
            led_worker();
        }

        // IO expander worker
        expander_worker();

        // Touch screen worker
        if(!record){
            if (touch_worker(&touch_x1, &touch_y1) == E_NO_ERROR) {

                // Check if init page start button is clicked
                if (!device_settings.enable_max78000_video) {
                    if ((LCD_START_BUTTON_X1 <= touch_x1) && (touch_x1 <= LCD_START_BUTTON_X2) &&
                        (LCD_START_BUTTON_Y1 <= touch_y1) && (touch_y1 <= LCD_START_BUTTON_Y2)) {
                        device_settings.enable_max78000_video = 1;
                        qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_ENABLE_CMD);
                        PR_INFO("start button clicked");
                        // Start video
                        MXC_Delay(MXC_DELAY_MSEC(1000));
                        device_settings.enable_max78000_video_cnn = 1;
                    }
                }

                PR_INFO("touch %d %d", touch_x1, touch_y1);
                timestamps.activity_detected = timer_ms_tick;
            }
        }

        // Button worker
        if(record && getting_name){
            for (int try = 0; try < 3; try++) {
                qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_GETTING_NAME_EN);
                qspi_master_wait_video_int();
            }
            
            lcd_drawImage(lcd_data.buffer);
            get_name(&embeddings);
            for (int try = 0; try < 3; try++) {
                qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_GETTING_NAME_DSB);
                qspi_master_wait_video_int();
            }
            getting_name = 0;
                        
            printf("name:%s\n",embeddings.embeddings_name);
            MXC_TS_RemoveAllButton();
        }
        if(record && !capture && !getting_name){
            block_button_x = 0; //Enable exit button
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "RECORD MODE");
            fonts_putStringCentered(150, lcd_string_buff, &Font_11x18, RED, lcd_data.buffer);
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "CAPTURE");
            fonts_putStringCentered(200, lcd_string_buff, &Font_16x26, RED,lcd_data.buffer);

            MXC_TS_AddButton(1,180,240,240,1);
            lcd_data.refresh_screen = 1;
            key = MXC_TS_GetKey();
            if(key==1){
                for (int try = 0; try < 3; try++) {
                    qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_CAPTURE_EN);
                    qspi_master_wait_video_int();
                }
                capture = 1;
                PR_INFO("capture started");
            }
            MXC_TS_RemoveAllButton();
        }
        
        if(record && capture){
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "OK");
            fonts_putString(35, 200, lcd_string_buff, &Font_16x26, GREEN, 0, 0, lcd_data.buffer);
            MXC_TS_AddButton(15,180,110,240,2);
            snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "RETRY");
            fonts_putString(135, 200, lcd_string_buff, &Font_16x26, RED,0,0,lcd_data.buffer);
            MXC_TS_AddButton(135,180,225,240,3);
            lcd_data.refresh_screen = 1;
            key = MXC_TS_GetKey();
            if(key==2){
                for (int try = 0; try < 3; try++) {
                    qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_CAPTURE_ACCEPT);
                    qspi_master_wait_video_int();
                }
                PR_INFO("capture accepted")
                
            }
            if(key==3){
                for (int try = 0; try < 3; try++) {
                    qspi_master_send_video(NULL, 0, QSPI_PACKET_TYPE_VIDEO_CAPTURE_DISCARD);
                    qspi_master_wait_video_int();
                }
                capture = 0;
                PR_INFO("capture discarded")
                
            }
            MXC_TS_RemoveAllButton();
                      
        }
        if(!record && (embeddings.capture_number!=0)){
            embeddings.capture_number =0;
            getting_name = 1;
            find_names_number(names,&db_number);
        }         
        button_worker();       

        
        if (lcd_data.refresh_screen && device_settings.enable_lcd ) {
            while(spi_dma_busy_flag(MAX32666_LCD_DMA_CHANNEL)); //Wait for the dma
            refresh_screen();
            PR_INFO("LCD refresh_time: %d ms", timer_ms_tick - refresh_time);
            refresh_time = timer_ms_tick;
        }

        // Sleep until an interrupt
        __WFI();
         PR_INFO("Loop time: %d ms", timer_ms_tick - loop_time);
    }
}

static int refresh_screen(void)
{
    if (device_status.fuel_gauge_working) {
        snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "%3d%%", device_status.statistics.battery_soc);
        if (device_status.usb_chgin) {
            fonts_putString(LCD_WIDTH - 31, 3, lcd_string_buff, &Font_7x10, ORANGE, 0, 0, lcd_data.buffer);
        } else if (device_status.statistics.battery_soc <= MAX32666_SOC_WARNING_LEVEL) {
            fonts_putString(LCD_WIDTH - 31, 3, lcd_string_buff, &Font_7x10, RED, 0, 0, lcd_data.buffer);
        } else {
            fonts_putString(LCD_WIDTH - 31, 3, lcd_string_buff, &Font_7x10, GREEN, 0, 0, lcd_data.buffer);
        }
    }

    
  
    // Draw FaceID frame and result
    if (device_settings.enable_max78000_video && device_settings.enable_max78000_video_cnn) {
        if (device_status.classification_video.classification != CLASSIFICATION_NOTHING) {
            if(device_status.classification_video.classification != CLASSIFICATION_UNKNOWN){
                if(!record){
                    strncpy(lcd_string_buff, names[device_status.classification_video.max_embed_index], sizeof(lcd_string_buff) - 1);
                    fonts_putStringCentered(LCD_HEIGHT - 29, lcd_string_buff, &Font_16x26, video_string_color, lcd_data.buffer);
                }
            }
            else{
                if(!record){
                    strncpy(lcd_string_buff, "UNKNOWN", sizeof(lcd_string_buff) - 1);
                    fonts_putStringCentered(LCD_HEIGHT - 29, lcd_string_buff, &Font_16x26, video_string_color, lcd_data.buffer);    
                }            
            }
        }
    }


    if (device_settings.enable_lcd_statistics) {

        //Not available for the faceID demo
    }

    // Draw button in init screen
    if (device_settings.enable_max78000_video == 0) {
        // Start button
        fonts_drawFilledRectangle(LCD_START_BUTTON_X1, LCD_START_BUTTON_Y1, LCD_START_BUTTON_X2 - LCD_START_BUTTON_X1,
                                  LCD_START_BUTTON_Y2 - LCD_START_BUTTON_Y1, LGRAY, lcd_data.buffer);
        fonts_drawThickRectangle(LCD_START_BUTTON_X1, LCD_START_BUTTON_Y1, LCD_START_BUTTON_X2, LCD_START_BUTTON_Y2, LIGHTBLUE, 4, lcd_data.buffer);
        snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "Start Video");
        fonts_putStringCentered(LCD_START_BUTTON_Y1 + 10, lcd_string_buff, &Font_16x26, ADIBLUE, lcd_data.buffer);
    }

    if (device_settings.enable_max78000_audio) {
        if ((timestamps.screen_drew - timestamps.audio_result_received) < LCD_CLASSIFICATION_DURATION) {
            if (device_settings.enable_lcd_probabilty) {
                snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "%s (%d%%)",
                        device_status.classification_audio.result, (uint8_t) device_status.classification_audio.probability);
            } else {
                strncpy(lcd_string_buff, device_status.classification_audio.result, sizeof(lcd_string_buff) - 1);
            }
        }
    }
    //Not available for the faceID demo
    /*else {
        snprintf(lcd_string_buff, sizeof(lcd_string_buff) - 1, "Audio disabled");
        fonts_putStringCentered(3, lcd_string_buff, &Font_11x18, RED, lcd_data.buffer);
    }*/

    if ((timestamps.screen_drew - timestamps.notification_received) < LCD_NOTIFICATION_DURATION) {
        if (strlen(lcd_data.notification) < (LCD_WIDTH / Font_11x18.width)) {
            fonts_putStringCentered(LCD_HEIGHT - Font_11x18.height - 3, lcd_data.notification, &Font_11x18, lcd_data.notification_color, lcd_data.buffer);
        } else {
            fonts_putStringCentered(LCD_HEIGHT - Font_7x10.height - 3, lcd_data.notification, &Font_7x10, lcd_data.notification_color, lcd_data.buffer);
        }
    }

    if (lcd_drawImage(lcd_data.buffer) == E_NO_ERROR) {
        device_status.statistics.lcd_fps = (float) 1000.0 / (float)(timer_ms_tick - timestamps.screen_drew);
        timestamps.screen_drew = timer_ms_tick;
    }

    return E_NO_ERROR;
}

// Similar to Core 0, the entry point for Core 1
// is Core1Main()
// Execution begins when the CPU1 Clock is enabled
int Core1_Main(void)
{
    //  __asm__("BKPT");

    // Disable Core1 ICC1 Instruction cache
    core1_icc(0);

//    int ret = 0;

    PR_INFO("maxrefdes178_max32666 core1");

    core1_irq_init();

//    ret = ble_init();
//    if (ret != E_NO_ERROR) {
//        PR_ERROR("ble_init %d", ret);
//    }

    core1_init_done = 1;

    PR_INFO("core 1 init completed");

//    core1_icc(1);

    while (1) {
//        ble_worker();
    }

    return E_NO_ERROR;
}

static void core0_irq_init(void)
{
    // Disable all interrupts used by core1
//    NVIC_DisableIRQ(SysTick_IRQn);

    NVIC_DisableIRQ(BTLE_TX_DONE_IRQn);
    NVIC_DisableIRQ(BTLE_RX_RCVD_IRQn);
    NVIC_DisableIRQ(BTLE_RX_ENG_DET_IRQn);
    NVIC_DisableIRQ(BTLE_SFD_DET_IRQn);
    NVIC_DisableIRQ(BTLE_SFD_TO_IRQn);
    NVIC_DisableIRQ(BTLE_GP_EVENT_IRQn);
    NVIC_DisableIRQ(BTLE_CFO_IRQn);
    NVIC_DisableIRQ(BTLE_SIG_DET_IRQn);
    NVIC_DisableIRQ(BTLE_AGC_EVENT_IRQn); // Disabled
    NVIC_DisableIRQ(BTLE_RFFE_SPIM_IRQn);
    NVIC_DisableIRQ(BTLE_TX_AES_IRQn); // Disabled
    NVIC_DisableIRQ(BTLE_RX_AES_IRQn); // Disabled
    NVIC_DisableIRQ(BTLE_INV_APB_ADDR_IRQn); // Disabled
    NVIC_DisableIRQ(BTLE_IQ_DATA_VALID_IRQn); // Disabled

    NVIC_DisableIRQ(MXC_TMR_GET_IRQ(MXC_TMR_GET_IDX(MAX32666_TIMER_BLE)));
    NVIC_DisableIRQ(MXC_TMR_GET_IRQ(MXC_TMR_GET_IDX(MAX32666_TIMER_BLE_SLEEP)));

    NVIC_DisableIRQ(WUT_IRQn);
}

static void core1_irq_init(void)
{
//    NVIC_DisableIRQ(SysTick_IRQn);

    NVIC_DisableIRQ(GPIO0_IRQn);
    NVIC_DisableIRQ(GPIO1_IRQn);

    NVIC_DisableIRQ(MXC_TMR_GET_IRQ(MXC_TMR_GET_IDX(MAX32666_TIMER_MS)));
}

static void core0_icc(int enable)
{
    MXC_ICC0->invalidate = 1;
    while (!(MXC_ICC0->cache_ctrl & MXC_F_ICC_CACHE_CTRL_RDY));
    if (enable) {
        MXC_ICC0->cache_ctrl |= MXC_F_ICC_CACHE_CTRL_EN;
    } else {
        MXC_ICC0->cache_ctrl &= ~MXC_F_ICC_CACHE_CTRL_EN;
    }
    while (!(MXC_ICC0->cache_ctrl & MXC_F_ICC_CACHE_CTRL_RDY));
}

static void core1_icc(int enable)
{
    MXC_ICC1->invalidate = 1;
    while (!(MXC_ICC1->cache_ctrl & MXC_F_ICC_CACHE_CTRL_RDY));
    if (enable) {
        MXC_ICC1->cache_ctrl |= MXC_F_ICC_CACHE_CTRL_EN;
    } else {
        MXC_ICC1->cache_ctrl &= ~MXC_F_ICC_CACHE_CTRL_EN;
    }
    while (!(MXC_ICC1->cache_ctrl & MXC_F_ICC_CACHE_CTRL_RDY));
}

static void digit_detection_result(void)
{
uint8_t *ml_data8 = lcd_data.ml_data8;
uint16_t color;
uint8_t class_idx, x1, x2, y1, y2;

    if(ml_data8[0] > 10)
    {
        PR_ERROR("More than ten objects!");
        return;
    }

    for (int i = 0; i < ml_data8[0]; i++) {

        // Extract info of detected digits and box coordinates
        class_idx = ml_data8[ML_DATA_SIZE*i + 1];
        x1 = ml_data8[ML_DATA_SIZE*i + 2];
        y1 = ml_data8[ML_DATA_SIZE*i + 3];
        x2 = ml_data8[ML_DATA_SIZE*i + 4];
        y2 = ml_data8[ML_DATA_SIZE*i + 5];

        if (x1 > LCD_WIDTH) {
            x1 = LCD_WIDTH;
            PR_ERROR("x1 > LCD_WIDTH");
        }
        
        if (x2 > LCD_WIDTH) {
            x2 = LCD_WIDTH;
            PR_ERROR("x2 > LCD_WIDTH");
        }
        
        if (y1 > LCD_HEIGHT) {
            y1 = LCD_HEIGHT;
            PR_ERROR("y1 > LCD_HEIGHT");
        }
        
        if (y2 > LCD_HEIGHT) {
            y2 = LCD_HEIGHT;
            PR_ERROR("y2 > LCD_HEIGHT");
        }

        // Set color according to detected digit
        if (class_idx == 0) {
            color = BLUE;
        } else if (class_idx == 1) {
            color = RED;
        } else if (class_idx == 2) {
            color = MAGENTA;
        } else if (class_idx == 3) {
            color = GREEN;
        } else if (class_idx == 4) {
            color = CYAN;
        } else if (class_idx == 5) {
            color = YELLOW;
        } else if (class_idx == 6) {
            color = ORANGE;
        } else if (class_idx == 7) {
            color = GREEN;
        } else if (class_idx == 8) {
            color = MAGENTA;
        } else if (class_idx == 9) {
            color = BLUE;
        }
        else {   
            return;
        }

        PR_DEBUG("Class:%d Color:%x", class_idx, color);
        PR_DEBUG("x1=%d, y1=%d, x2=%d, y2=%d", x1, y1, x2, y2);

        // Draw rectangle around detected digit
        fonts_drawThickRectangle(x1, y1, x2, y2, color, BOX_THICKNESS, lcd_data.buffer);
        
        memset(lcd_string_buff, 0xff, sizeof(lcd_string_buff));
        snprintf(lcd_string_buff, label_text[class_idx].len + 1, label_text[class_idx].data);
        // Show detected digit
        fonts_putString(x1+BOX_THICKNESS, y1+BOX_THICKNESS, lcd_string_buff, &Font_16x26, GREEN, 0, 0, lcd_data.buffer);
        PR_DEBUG("Face Detected!");
        if(record){
            if(capture){
                cnn_2_load_weights_from_SD();
            }
        }
        else{
            cnn_2_load_weights_from_SD();
        }

    }
}
