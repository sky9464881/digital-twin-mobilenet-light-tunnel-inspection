/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* =========================================================
   사용자가 나중에 조정할 가능성이 큰 값들
   ========================================================= */

/* 버튼은 PB12 + Pull-down 이므로
   버튼을 누르면 HIGH가 들어온다고 가정 */
#define START_SW_ACTIVE_LEVEL      GPIO_PIN_SET

/* 버튼 디바운스 시간 */
#define START_SW_DEBOUNCE_MS       30U

/* 스텝모터 Enable 핀 active level
   A4988 / DRV8825 계열은 보통 EN이 LOW일 때 활성인 경우가 많음
   모터가 enable/disable 반대로 동작하면 이 두 줄을 서로 바꾸면 됨 */
#define STEP_EN_ACTIVE_LEVEL       GPIO_PIN_RESET
#define STEP_EN_INACTIVE_LEVEL     GPIO_PIN_SET

/* 모터 이동 방향
   실제 회전 방향이 반대로 나오면 SET/RESET을 서로 바꾸면 됨 */
#define MOVE_IN_DIR                GPIO_PIN_SET
#define MOVE_OUT_DIR               GPIO_PIN_RESET

/* 이동 step 수
   현재는 예시값
   1.8도 모터 + 풀스텝이면 90도 = 50 step */
#define MOVE_IN_STEPS              11500U
#define MOVE_OUT_STEPS             7000U

/* 시간 조건 */
#define SETTLE_DELAY_MS            1000U   /* 모터 1차 이동 후 안정화 대기 */
#define LED_PRE_DELAY_MS           5000U   /* LED ON 후 트리거까지 대기 */
#define LED_POST_DELAY_MS          5000U   /* 트리거 후 LED OFF까지 대기 */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;

/* USER CODE BEGIN PV */

/* 현재 시퀀스가 동작 중인지 표시
   0이면 대기 상태, 1이면 동작 중 */
static uint8_t busy = 0U;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM2_Init(void);
/* USER CODE BEGIN PFP */
static uint8_t button_is_pressed(void);
static void wait_button_release(void);

static void all_led_off(void);

static void stepper_enable(void);
static void stepper_disable(void);
static void stepper_move_steps(uint32_t steps, GPIO_PinState dir);

static void camera_trigger_once(void);
static void capture_with_led(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);

static void run_sequence(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ---------------------------------------------------------
   버튼이 눌렸는지 확인
   START_SW_ACTIVE_LEVEL 기준으로 눌림 여부 판단
   --------------------------------------------------------- */
static uint8_t button_is_pressed(void)
{
    return (HAL_GPIO_ReadPin(START_SW_GPIO_Port, START_SW_Pin) == START_SW_ACTIVE_LEVEL);
}

/* ---------------------------------------------------------
   버튼에서 손을 뗄 때까지 기다림
   버튼을 계속 누르고 있을 때 중복 실행되는 것 방지
   --------------------------------------------------------- */
static void wait_button_release(void)
{
    while (button_is_pressed())
    {
        HAL_Delay(1);
    }

    /* 손을 뗀 직후 채터링 방지용 짧은 대기 */
    HAL_Delay(START_SW_DEBOUNCE_MS);
}

/* ---------------------------------------------------------
   LED 전체 OFF
   --------------------------------------------------------- */
static void all_led_off(void)
{
    HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED2_GPIO_Port, LED2_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED3_GPIO_Port, LED3_Pin, GPIO_PIN_RESET);
}

/* ---------------------------------------------------------
   스텝모터 enable
   --------------------------------------------------------- */
static void stepper_enable(void)
{
    HAL_GPIO_WritePin(STEP_EN_GPIO_Port, STEP_EN_Pin, STEP_EN_ACTIVE_LEVEL);
}

/* ---------------------------------------------------------
   스텝모터 disable
   --------------------------------------------------------- */
static void stepper_disable(void)
{
    HAL_GPIO_WritePin(STEP_EN_GPIO_Port, STEP_EN_Pin, STEP_EN_INACTIVE_LEVEL);
}

/* ---------------------------------------------------------
   stepper_move_steps()
   - TIM1 CH1(STEP_PULSE)에서 STEP 펄스를 발생
   - update event 1개를 step 1개로 계산
   - 방향은 STEP_DIR 핀으로 제어
   --------------------------------------------------------- */
static void stepper_move_steps(uint32_t steps, GPIO_PinState dir)
{
    /* 먼저 회전 방향 설정 */
    HAL_GPIO_WritePin(STEP_DIR_GPIO_Port, STEP_DIR_Pin, dir);

    /* 타이머 카운터와 update flag 초기화 */
    __HAL_TIM_SET_COUNTER(&htim1, 0);
    __HAL_TIM_CLEAR_FLAG(&htim1, TIM_FLAG_UPDATE);

    /* STEP PWM 시작 */
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);

    /* 목표 step 수만큼 타이머 주기를 기다림
       현재 TIM1 설정(PSC=15, ARR=999)이면
       HSI 16MHz 기준 대략 1kHz step 펄스가 나옴 */
    for (uint32_t i = 0; i < steps; i++)
    {
        while (__HAL_TIM_GET_FLAG(&htim1, TIM_FLAG_UPDATE) == RESET)
        {
            /* 한 주기 끝날 때까지 대기 */
        }

        /* 다음 step을 세기 위해 flag 클리어 */
        __HAL_TIM_CLEAR_FLAG(&htim1, TIM_FLAG_UPDATE);
    }

    /* 목표 step 수 도달 -> PWM 정지 */
    HAL_TIM_PWM_Stop(&htim1, TIM_CHANNEL_1);

    /* 모터 정지 후 아주 짧게 안정화 */
    HAL_Delay(10);
}

/* ---------------------------------------------------------
   camera_trigger_once()
   - TIM2 CH1(CAM_TRIG)에서 카메라 트리거 펄스를 1번 출력
   - PWM을 1주기만 내보내고 정지
   --------------------------------------------------------- */
static void camera_trigger_once(void)
{
	HAL_GPIO_WritePin(CAM_TRIG_GPIO_Port, CAM_TRIG_Pin, GPIO_PIN_RESET);
	HAL_Delay(10);
	HAL_GPIO_WritePin(CAM_TRIG_GPIO_Port, CAM_TRIG_Pin, GPIO_PIN_SET);
	HAL_Delay(1500);
	HAL_GPIO_WritePin(CAM_TRIG_GPIO_Port, CAM_TRIG_Pin, GPIO_PIN_RESET);
}

/* ---------------------------------------------------------
   capture_with_led()
   각 LED마다 아래 순서 실행
   1) LED ON
   2) 1초 대기
   3) 카메라 트리거 1회
   4) 1초 대기
   5) LED OFF
   --------------------------------------------------------- */
static void capture_with_led(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
    /* LED 켜기 */
    HAL_GPIO_WritePin(GPIOx, GPIO_Pin, GPIO_PIN_SET);

    /* LED 켠 뒤 1초 대기 */
    HAL_Delay(LED_PRE_DELAY_MS);

    /* 카메라 트리거 1회 */
    camera_trigger_once();

    /* 트리거 후 1초 대기 */
    HAL_Delay(LED_POST_DELAY_MS);

    /* LED 끄기 */
    HAL_GPIO_WritePin(GPIOx, GPIO_Pin, GPIO_PIN_RESET);
}

/* ---------------------------------------------------------
   run_sequence()
   전체 동작 순서
   1) 모터 enable
   2) 촬영 위치까지 이동
   3) 1초 대기
   4) LED1 촬영
   5) LED2 촬영
   6) LED3 촬영
   7) 다시 이동
   8) 모터 disable
   9) LED 전체 OFF
   --------------------------------------------------------- */
static void run_sequence(void)
{
    busy = 1U;

    /* 시작 전에 LED 모두 OFF */
    all_led_off();


    /* 모터 활성화 */
    stepper_enable();
    /* 1차 이동: 촬영 위치로 이동 */
    stepper_move_steps(MOVE_IN_STEPS, MOVE_OUT_DIR);
    /* 모터 정지 후 1초 대기 */
    HAL_Delay(SETTLE_DELAY_MS);



    /* LED1 촬영 */
    capture_with_led(LED1_GPIO_Port, LED1_Pin);

    /* LED2 촬영 */
    capture_with_led(LED2_GPIO_Port, LED2_Pin);

    /* LED3 촬영 */
    capture_with_led(LED3_GPIO_Port, LED3_Pin);





    /* 2차 이동: 배출 또는 원위치 */
    stepper_move_steps(MOVE_OUT_STEPS, MOVE_OUT_DIR);

    /* 종료 처리 */
    stepper_disable();
    all_led_off();

    busy = 0U;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */

  /* 전원 인가 직후 초기 상태 정리 */
  all_led_off();
  stepper_disable();

  /* 방향 핀 기본값 */
  HAL_GPIO_WritePin(STEP_DIR_GPIO_Port, STEP_DIR_Pin, GPIO_PIN_RESET);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
      /* 버튼이 눌렸고 현재 동작 중이 아니면 */
      if ((busy == 0U) && button_is_pressed())
      {
          /* 디바운스 */
          HAL_Delay(START_SW_DEBOUNCE_MS);

          /* 여전히 눌려 있으면 유효 입력으로 판단 */
          if (button_is_pressed())
          {
              /* 버튼에서 손을 뗄 때까지 기다린 뒤 시퀀스 실행 */
              wait_button_release();
              run_sequence();
          }
      }

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 15;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 999;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 500;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 15;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 999;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(STEP_DIR_GPIO_Port, STEP_DIR_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0|STEP_EN_Pin|LED1_Pin|LED2_Pin
                          |LED3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : STEP_DIR_Pin */
  GPIO_InitStruct.Pin = STEP_DIR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(STEP_DIR_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PA0 STEP_EN_Pin LED1_Pin LED2_Pin
                           LED3_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_0|STEP_EN_Pin|LED1_Pin|LED2_Pin
                          |LED3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : START_SW_Pin */
  GPIO_InitStruct.Pin = START_SW_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(START_SW_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
