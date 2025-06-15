#include <cstring>
#include "display/lcd_display.h"
#include <esp_log.h>
#include "mmap_generate_emoji.h"
// #include "esp_mmap_assets.h"
#include "emoji_display.h"
#include "assets/lang_config.h"
#include "config.h"
#include "font_awesome_symbols.h"
#include <esp_lvgl_port.h>

#include <esp_lcd_panel_io.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/event_groups.h>

static const char *TAG = "emoji";
LV_FONT_DECLARE(font_puhui_30_4);
LV_FONT_DECLARE(font_awesome_20_4);

namespace anim {

bool EmojiPlayer::OnFlushIoReady(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_io_event_data_t *edata, void *user_ctx)
{
    auto* disp_drv = static_cast<anim_player_handle_t*>(user_ctx);
    anim_player_flush_ready(disp_drv);
    return true;
}

void EmojiPlayer::OnFlush(anim_player_handle_t handle, int x_start, int y_start, int x_end, int y_end, const void *color_data)
{
    auto* panel = static_cast<esp_lcd_panel_handle_t>(anim_player_get_user_data(handle));
    esp_lcd_panel_draw_bitmap(panel, x_start, y_start, x_end, y_end, color_data);
}

EmojiPlayer::EmojiPlayer(esp_lcd_panel_handle_t panel, esp_lcd_panel_io_handle_t panel_io)
{
    ESP_LOGI(TAG, "Create EmojiPlayer, panel: %p, panel_io: %p", panel, panel_io);
    const mmap_assets_config_t assets_cfg = {
        .partition_label = "assets_A",
        .max_files = MMAP_EMOJI_FILES,
        .checksum = MMAP_EMOJI_CHECKSUM,
        .flags = {.mmap_enable = true, .full_check = true}
    };

    mmap_assets_new(&assets_cfg, &assets_handle_);

    anim_player_config_t player_cfg = {
        .flush_cb = OnFlush,
        .update_cb = NULL,
        .user_data = panel,
        .flags = {.swap = true},
        .task = ANIM_PLAYER_INIT_CONFIG()
    };

    player_handle_ = anim_player_init(&player_cfg);

    const esp_lcd_panel_io_callbacks_t cbs = {
        .on_color_trans_done = OnFlushIoReady,
    };
    esp_lcd_panel_io_register_event_callbacks(panel_io, &cbs, player_handle_);
    StartPlayer(MMAP_EMOJI_CONNECTING_AAF, true, 10);
}

EmojiPlayer::~EmojiPlayer()
{
    if (player_handle_) {
        anim_player_update(player_handle_, PLAYER_ACTION_STOP);
        anim_player_deinit(player_handle_);
        player_handle_ = nullptr;
    }

    if (assets_handle_) {
        mmap_assets_del(assets_handle_);
        assets_handle_ = NULL;
    }
}

void EmojiPlayer::StartPlayer(int aaf, bool repeat, int fps)
{
    if (player_handle_) {
        uint32_t start, end;
        const void *src_data;
        size_t src_len;

        src_data = mmap_assets_get_mem(assets_handle_, aaf);
        src_len = mmap_assets_get_size(assets_handle_, aaf);

        anim_player_set_src_data(player_handle_, src_data, src_len);
        anim_player_get_segment(player_handle_, &start, &end);
        // if(MMAP_EMOJI_WAKE_AAF == aaf){
        //     start = 7;
        // }
        anim_player_set_segment(player_handle_, start, end, fps, true);
        anim_player_update(player_handle_, PLAYER_ACTION_START);
    }
}

void EmojiPlayer::StopPlayer()
{
    if (player_handle_) {
        anim_player_update(player_handle_, PLAYER_ACTION_STOP);
    }
}

EmojiWidget::EmojiWidget(esp_lcd_panel_handle_t panel, 
                        esp_lcd_panel_io_handle_t panel_io)
                : SpiLcdDisplay(panel_io, panel, DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_OFFSET_X,DISPLAY_OFFSET_Y, DISPLAY_MIRROR_X, true, DISPLAY_SWAP_XY,
                    {
                        .text_font = &font_puhui_30_4,
                        .icon_font = &font_awesome_20_4,
                        .emoji_font = font_emoji_64_init(),
                    }) 
{
    this->panel_ = panel;
    // this->panel_num_ = 0; // 0: panel, 1: panel_pic
    InitializePlayer(panel, panel_io);
}

EmojiWidget::~EmojiWidget()
{

}

void EmojiWidget::SetEmotion(const char* emotion)
{
    if (!player_) {
        ESP_LOGE(TAG, "SetEmotion player_ is null, restarting player");   
        player_->StartPlayer(MMAP_EMOJI_NEUTRAL_AAF, true, 1);
        return;
    }

    using Param = std::tuple<int, bool, int>;
    static const std::unordered_map<std::string, Param> emotion_map = {
        {"happy",       {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 15}},
        {"laughing",    {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 20}},
        {"funny",       {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 15}},
        {"loving",      {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 15}},
        {"embarrassed", {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 25}},
        {"confident",   {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 25}},
        {"delicious",   {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 25}},
        {"sad",         {MMAP_EMOJI_SAD_LOOP_AAF,   true, 10}},
        {"crying",      {MMAP_EMOJI_SAD_LOOP_AAF,   true, 15}},
        {"sleepy",      {MMAP_EMOJI_SLEEP_AAF,      true, 1}},
        {"silly",       {MMAP_EMOJI_SAD_LOOP_AAF,   true, 15}},
        {"angry",       {MMAP_EMOJI_ANGER_LOOP_AAF, true, 25}},
        {"surprised",   {MMAP_EMOJI_PANIC_LOOP_AAF, true, 25}},
        {"shocked",     {MMAP_EMOJI_PANIC_LOOP_AAF, true, 25}},
        {"thinking",    {MMAP_EMOJI_HAPPY_LOOP_AAF, true, 15}},
        {"winking",     {MMAP_EMOJI_BLINK_QUICK_AAF, true, 10}},
        {"relaxed",     {MMAP_EMOJI_SCORN_LOOP_AAF, true, 5}},
        {"confused",    {MMAP_EMOJI_SCORN_LOOP_AAF, true, 5}},
        {"wake",        {MMAP_EMOJI_WAKE_AAF,       true, 15}},
        {"neutral",     {MMAP_EMOJI_NEUTRAL_AAF,    true, 1}},
        {"connecting",  {MMAP_EMOJI_CONNECTING_AAF, true, 1}},
        {"ask",         {MMAP_EMOJI_ASKING_AAF,     true, 10}},
    };

    auto it = emotion_map.find(emotion);
    if (it != emotion_map.end()) {
        const auto& [aaf, repeat, fps] = it->second;
        player_->StartPlayer(aaf, repeat, fps);
    } else if (strcmp(emotion, "neutral") == 0) {
    }
}

void EmojiWidget::SetStatus(const char* status)
{
    if(!player_) {
        ESP_LOGE(TAG, "SetStatus player_ is null, restarting player");   
        player_->StartPlayer(MMAP_EMOJI_NEUTRAL_AAF, true, 1);
        // return;
    }

    if (player_) {
        if (strcmp(status, Lang::Strings::LISTENING) == 0) {
            player_->StartPlayer(MMAP_EMOJI_ASKING_AAF, true, 15);
        } else if (strcmp(status,Lang::Strings::STANDBY) == 0) {
            player_->StartPlayer(MMAP_EMOJI_WAKE_AAF, true, 15);
        }
    }
}

void EmojiWidget::SetPreviewImage(const lv_img_dsc_t* img_dsc)
{
    ESP_LOGI(TAG, "SetPreviewImage called with img_dsc: %p", img_dsc);
    if(preview_image_ == nullptr) {
        ESP_LOGE(TAG, "preview_image_ is null, cannot set preview image");
        return;
    }

    DisplayLockGuard lock(this);
    if (img_dsc != nullptr) {

        // stop player if it is running
        if (player_) {
            player_->StopPlayer();
            ESP_LOGI(TAG, "StopPlayer before setting preview image");
        }
        //zuki :zoom 不缩放  缩小至 50% (128/256 = 0.5)
        lv_image_set_scale(preview_image_, 256);
        lv_img_set_src(preview_image_, img_dsc);
        lv_obj_clear_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);

        //去掉 anim player
        // player_ = nullptr;

    } else {
        lv_obj_add_flag(preview_image_, LV_OBJ_FLAG_HIDDEN);

        //重新启动player
        if (player_) {
            ESP_LOGI(TAG, "ReStart Player after preview image");
            player_->StartPlayer(MMAP_EMOJI_NEUTRAL_AAF, true, 1);
        }
    }

}


void EmojiWidget::InitializePlayer(esp_lcd_panel_handle_t panel, esp_lcd_panel_io_handle_t panel_io)
{
    player_ = std::make_unique<EmojiPlayer>(panel, panel_io);
}

bool EmojiWidget::Lock(int timeout_ms)
{
    return true;
}

void EmojiWidget::Unlock()
{
}

} // namespace anim
