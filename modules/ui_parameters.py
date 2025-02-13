from pathlib import Path 

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui(default_preset):
    mu = shared.args.multi_user
    generate_params = presets.load_preset(default_preset)
    with gr.Tab("Parameters", elem_id="parameters"):
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(
                            choices=utils.get_available_presets(), 
                            value=default_preset, 
                            label='Preset', 
                            elem_classes='slim-dropdown'
                        )
                        ui.create_refresh_button(
                            shared.gradio['preset_menu'], 
                            lambda: None, 
                            lambda: {'choices': utils.get_available_presets()}, 
                            'refresh-button', 
                            interactive=not mu
                        )
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['random_preset'] = gr.Button('🎲', elem_classes='refresh-button')
                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(
                        label="Filter by loader", 
                        choices=["All"] + list(loaders.loaders_and_params.keys()), 
                        value="All", 
                        elem_classes='slim-dropdown'
                    )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('## Curve shape')
                            shared.gradio['temperature'] = gr.Slider(0.01, 5, value=generate_params['temperature'], step=0.01, label='temperature')
                            shared.gradio['dynatemp_low'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_low'], step=0.01, label='dynatemp_low', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_high'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_high'], step=0.01, label='dynatemp_high', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_exponent'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_exponent'], step=0.01, label='dynatemp_exponent', visible=generate_params['dynamic_temperature'])
                            shared.gradio['smoothing_factor'] = gr.Slider(0.0, 10.0, value=generate_params['smoothing_factor'], step=0.01, label='smoothing_factor', info='Activates Quadratic Sampling.')
                            shared.gradio['smoothing_curve'] = gr.Slider(1.0, 10.0, value=generate_params['smoothing_curve'], step=0.01, label='smoothing_curve', info='Adjusts the dropoff curve of Quadratic Sampling.')
                            
                            gr.Markdown('## Curve cutoff')
                            shared.gradio['min_p'] = gr.Slider(0.0, 1.0, value=generate_params['min_p'], step=0.01, label='min_p')
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p')
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k')
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p')
                            shared.gradio['xtc_threshold'] = gr.Slider(0, 0.5, value=generate_params['xtc_threshold'], step=0.01, label='xtc_threshold', info='If 2 or more tokens have probability above this threshold, consider removing all but the last one.')
                            shared.gradio['xtc_probability'] = gr.Slider(0, 1, value=generate_params['xtc_probability'], step=0.01, label='xtc_probability', info='Probability that the removal will actually happen. 0 disables the sampler. 1 makes it always happen.')
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff')
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')
                            
                            gr.Markdown('## Repetition suppression')
                            shared.gradio['dry_multiplier'] = gr.Slider(0, 5, value=generate_params['dry_multiplier'], step=0.01, label='dry_multiplier', info='Set to greater than 0 to enable DRY. Recommended value: 0.8.')
                            shared.gradio['dry_allowed_length'] = gr.Slider(1, 20, value=generate_params['dry_allowed_length'], step=1, label='dry_allowed_length', info='Longest sequence that can be repeated without being penalized.')
                            shared.gradio['dry_base'] = gr.Slider(1, 4, value=generate_params['dry_base'], step=0.01, label='dry_base', info='Controls how fast the penalty grows with increasing sequence length.')
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty')
                            shared.gradio['frequency_penalty'] = gr.Slider(0, 2, value=generate_params['frequency_penalty'], step=0.05, label='frequency_penalty')
                            shared.gradio['presence_penalty'] = gr.Slider(0, 2, value=generate_params['presence_penalty'], step=0.05, label='presence_penalty')
                            shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
                            shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range')
                        with gr.Column():
                            gr.Markdown('## Alternative sampling methods')
                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='For Contrastive Search. do_sample must be unchecked.')
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='For CFG. 1.5 is a good value.')
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='mode=1 is for llama.cpp only.')
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')
                            
                            gr.Markdown('## Inference Time Extension (Chain-of-Thought Refinement)')
                            shared.gradio['cot_refinement'] = gr.Checkbox(
                                value=generate_params.get('cot_refinement', False), 
                                label='Enable CoT Refinement', 
                                info='透過多次梯度更新，使 logits 熵靠近目標值，延長模型「思考」時間。'
                            )
                            shared.gradio['cot_target_entropy'] = gr.Slider(
                                0.0, 2.0, 
                                value=generate_params.get('cot_target_entropy', 0.5), 
                                step=0.01, 
                                label='Target Entropy'
                            )
                            shared.gradio['cot_max_iter'] = gr.Slider(
                                1, 10, 
                                value=generate_params.get('cot_max_iter', 3), 
                                step=1, 
                                label='Max Iterations'
                            )
                            shared.gradio['cot_step_size'] = gr.Slider(
                                0.001, 0.2, 
                                value=generate_params.get('cot_step_size', 0.05), 
                                step=0.001, 
                                label='Step Size'
                            )
                            
                            gr.Markdown('## MCTS Sampling (內部化 MCTS 局部樹搜索)')
                            shared.gradio['mcts_enabled'] = gr.Checkbox(
                                value=generate_params.get('mcts_enabled', False), 
                                label='Enable MCTS Sampling', 
                                info='使用內部化 MCTS 進行局部樹搜索，提升 token 選擇的智慧。'
                            )
                            shared.gradio['mcts_candidate_count'] = gr.Slider(
                                1, 10, 
                                value=generate_params.get('mcts_candidate_count', 5), 
                                step=1, 
                                label='Candidate Count', 
                                info='每次從 logits 中取出候選 token 的數量。'
                            )
                            shared.gradio['mcts_rollout_depth'] = gr.Slider(
                                1, 5, 
                                value=generate_params.get('mcts_rollout_depth', 2), 
                                step=1, 
                                label='Rollout Depth', 
                                info='每個候選 token 進行 rollout 的深度。'
                            )
                            shared.gradio['mcts_rollout_samples'] = gr.Slider(
                                1, 5, 
                                value=generate_params.get('mcts_rollout_samples', 1), 
                                step=1, 
                                label='Rollout Samples', 
                                info='每個候選 token rollout 的次數。'
                            )
                            shared.gradio['mcts_exploration_const'] = gr.Slider(
                                0.1, 5.0, 
                                value=generate_params.get('mcts_exploration_const', 1.0), 
                                step=0.1, 
                                label='Exploration Constant', 
                                info='控制探索與利用之間的平衡。'
                            )
                            shared.gradio['mcts_context_window'] = gr.Slider(
                                1, 20, 
                                value=generate_params.get('mcts_context_window', 10), 
                                step=1, 
                                label='MCTS Context Window', 
                                info='從輸入中取最後多少個 token 作為上下文。'
                            )
                            shared.gradio['mcts_penalty_factor'] = gr.Slider(
                                0.0, 1.0, 
                                value=generate_params.get('mcts_penalty_factor', 0.1), 
                                step=0.01, 
                                label='MCTS Penalty Factor', 
                                info='上下文中重複 token 的懲罰因子。'
                            )
                            
                            # --- 新增：Enhanced Oscillatory Reflection Sampling ---
                            gr.Markdown('## Enhanced Oscillatory Reflection Sampling')
                            shared.gradio['enhanced_oscillatory_reflection'] = gr.Checkbox(
                                value=generate_params.get('enhanced_oscillatory_reflection', False),
                                label='Enable Enhanced Oscillatory Reflection Sampling',
                                info='啟用後將使用先進的振盪反饋採樣器，透過混沌映射、上下文自適應與隨機失活提升生成創意。'
                                    )
                            shared.gradio['enhanced_oscillatory_base_amplitude'] = gr.Slider(
                                minimum=0.0, maximum=2.0,
                                value=generate_params.get('enhanced_oscillatory_base_amplitude', 0.5),
                                step=0.01,
                                label='Enhanced Base Amplitude',
                                info='基本振幅，用於控制干擾強度。'
                                    )
                            shared.gradio['enhanced_oscillatory_base_frequency'] = gr.Slider(
                                minimum=0.0, maximum=2.0,
                                value=generate_params.get('enhanced_oscillatory_base_frequency', 0.1),
                                step=0.01,
                                label='Enhanced Base Frequency',
                                info='基本頻率，影響振盪的週期性。'
                                )
                            shared.gradio['enhanced_oscillatory_dropout_prob'] = gr.Slider(
                                minimum=0.0, maximum=1.0,
                                value=generate_params.get('enhanced_oscillatory_dropout_prob', 0.1),
                                step=0.01,
                                label='Enhanced Dropout Probability',
                                info='隨機失活的概率，可防止過度依賴干擾效果。'
                                    )

                            gr.Markdown('## HyperbolicFractal')
                            shared.gradio['hhflw_enabled'] = gr.Checkbox(
                                value=generate_params.get('hhflw_enabled', False), 
                                label='hhflw_enabled', 
                                info='Enable hhflw.'
                            )
                            shared.gradio['hflw_alpha'] = gr.Slider(
                                0.1, 5.0, 
                                value=generate_params.get('hflw_alpha', 1.0), 
                                step=0.1, 
                                label='hflw_alpha', 
                                info='Controls stability of high-probability tokens.g.'
                            )
                            shared.gradio['hflw_beta'] = gr.Slider(
                                0.0, 1.0, 
                                value=generate_params.get('hflw_beta', 0.5), 
                                step=0.01, 
                                label='hflw_beta', 
                                info='Controls scaling of lower-probability tokens.'
                            )
                            shared.gradio['hflw_p_init'] = gr.Slider(
                                0.1, 5.0, 
                                value=generate_params.get('hflw_p_init', 2.0), 
                                step=0.1, 
                                label='hflw_p_init', 
                                info='Initial exponent for dynamic scaling.'
                            )
                            shared.gradio['hflw_lambda_factor'] = gr.Slider(
                                0.0, 1.0, 
                                value=generate_params.get('hflw_lambda_factor', 0.1), 
                                step=0.01, 
                                label='hflw_lambda_factor', 
                                info='Learning rate for fractal adaptation.'
                            )

                            gr.Markdown('## Hypernomic_Gradient_Descent')
                            shared.gradio['hypernomic_gradient'] = gr.Checkbox(
                                value=generate_params.get('hypernomic_gradient', False), 
                                label='hypernomic_gradient', 
                                info='Enable hypernomic_gradient.'
                            )
                            shared.gradio['hgd_alpha'] = gr.Slider(
                                0.1, 3.0, 
                                value=generate_params.get('hgd_alpha', 1.2), 
                                step=0.1, 
                                label='hgd_alpha', 
                                info='Entropy modulation coefficient.'
                            )
                            shared.gradio['hgd_beta'] = gr.Slider(
                                0.0, 3.0, 
                                value=generate_params.get('hgd_beta', 0.8), 
                                step=0.01, 
                                label='hgd_beta', 
                                info='Coherence stabilization parameter.'
                            )
                            shared.gradio['asm_tau'] = gr.Slider(
                                0.0, 3.0, 
                                value=generate_params.get('asm_tau', 0.05), 
                                step=0.01, 
                                label='asm_tau', 
                                info='Stability threshold.'
                            )
                            shared.gradio['asm_lambda'] = gr.Slider(
                                0.0, 3.0, 
                                value=generate_params.get('asm_lambda', 0.7), 
                                step=0.01, 
                                label='asm_lambda', 
                                info='Low-probability correction.'
                            )
                            shared.gradio['asm_mu'] = gr.Slider(
                                0.0, 3.0, 
                                value=generate_params.get('asm_mu', 0.3), 
                                step=0.01, 
                                label='asm_mu', 
                                info='Probability suppression factor.'
                            )

                            gr.Markdown('## Adaptive Sampling')
                            shared.gradio['margin_adaptive'] = gr.Checkbox(
                                value=generate_params.get('margin_adaptive', False), 
                                label='Margin Adaptive Sampler', 
                                info='Enable margin-based dynamic temperature scaling.'
                            )
                            shared.gradio['margin_threshold'] = gr.Slider(
                                0.1, 5.0, 
                                value=generate_params.get('margin_threshold', 1.0), 
                                step=0.1, 
                                label='Margin Threshold', 
                                info='Threshold for margin clipping.'
                            )
                            shared.gradio['margin_min_factor'] = gr.Slider(
                                0.0, 1.0, 
                                value=generate_params.get('margin_min_factor', 0.5), 
                                step=0.01, 
                                label='Margin Minimum Factor', 
                                info='Minimum scaling factor when margin is zero.'
                            )

                            gr.Markdown('## inference_time_extension')
                            shared.gradio['inference_time_extension'] = gr.Checkbox(
                                value=generate_params.get('inference_time_extension', False), 
                                label='inference time extension', 
                                info='Enable inference_time_extension.'
                            )
                            shared.gradio['inference_max_delay'] = gr.Slider(
                                0.1, 3.0, 
                                value=generate_params.get('inference_max_delay', 1.0), 
                                step=0.1, 
                                label='inference_max_delay', 
                                info='inference_max_delay.'
                            )
                            shared.gradio['inference_entropy_threshold'] = gr.Slider(
                                0.0, 3.0, 
                                value=generate_params.get('inference_entropy_threshold', 0.5), 
                                step=0.01, 
                                label='inference_entropy_threshold', 
                                info='inference_entropy_threshold.'
                            )
                            shared.gradio['inference_confidence_threshold'] = gr.Slider(
                                0.0, 3.0, 
                                value=generate_params.get('inference_confidence_threshold', 0.5), 
                                step=0.01, 
                                label='inference_confidence_threshold', 
                                info='inference_confidence_threshold.'
                            )
                            
                            gr.Markdown('## Other options')
                            shared.gradio['max_new_tokens'] = gr.Slider(
                                minimum=shared.settings['max_new_tokens_min'], 
                                maximum=shared.settings['max_new_tokens_max'], 
                                value=shared.settings['max_new_tokens'], 
                                step=1, 
                                label='max_new_tokens', 
                                info='⚠️ Setting this too high can cause prompt truncation.'
                            )
                            shared.gradio['prompt_lookup_num_tokens'] = gr.Slider(
                                value=shared.settings['prompt_lookup_num_tokens'], 
                                minimum=0, 
                                maximum=10, 
                                step=1, 
                                label='prompt_lookup_num_tokens', 
                                info='Activates Prompt Lookup Decoding.'
                            )
                            shared.gradio['max_tokens_second'] = gr.Slider(
                                value=shared.settings['max_tokens_second'], 
                                minimum=0, 
                                maximum=20, 
                                step=1, 
                                label='Maximum tokens/second', 
                                info='To make text readable in real time.'
                            )
                            shared.gradio['max_updates_second'] = gr.Slider(
                                value=shared.settings['max_updates_second'], 
                                minimum=0, 
                                maximum=24, 
                                step=1, 
                                label='Maximum UI updates/second', 
                                info='Set this if you experience lag in the UI during streaming.'
                            )
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
                            shared.gradio['dynamic_temperature'] = gr.Checkbox(value=generate_params['dynamic_temperature'], label='dynamic_temperature')
                            shared.gradio['temperature_last'] = gr.Checkbox(
                                value=generate_params['temperature_last'], 
                                label='temperature_last', 
                                info='Moves temperature/dynamic temperature/quadratic sampling to the end of the sampler stack, ignoring their positions in "Sampler priority".'
                            )
                            shared.gradio['auto_max_new_tokens'] = gr.Checkbox(
                                value=shared.settings['auto_max_new_tokens'], 
                                label='auto_max_new_tokens', 
                                info='Expand max_new_tokens to the available context length.'
                            )
                            shared.gradio['ban_eos_token'] = gr.Checkbox(
                                value=shared.settings['ban_eos_token'], 
                                label='Ban the eos_token', 
                                info='Forces the model to never end the generation prematurely.'
                            )
                            shared.gradio['add_bos_token'] = gr.Checkbox(
                                value=shared.settings['add_bos_token'], 
                                label='Add the bos_token to the beginning of prompts', 
                                info='Disabling this can make the replies more creative.'
                            )
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(
                                value=shared.settings['skip_special_tokens'], 
                                label='Skip special tokens', 
                                info='Some specific models need this unset.'
                            )
                            shared.gradio['stream'] = gr.Checkbox(
                                value=shared.settings['stream'], 
                                label='Activate text streaming'
                            )
                            shared.gradio['static_cache'] = gr.Checkbox(
                                value=shared.settings['static_cache'], 
                                label='Static KV cache', 
                                info='Use a static cache for improved performance.'
                            )
        # 新增一個獨立的標籤頁來放置 PanopticConsciousnessSampler 的參數
                with gr.Tab("Panoptic Consciousness Sampler"):
                    gr.Markdown("## Panoptic Consciousness Sampler Parameters")
            
                    shared.gradio['panoptic_consciousness'] = gr.Checkbox(
                        value=generate_params.get('panoptic_consciousness', False),
                        label='Enable Panoptic Consciousness Sampling',
                        info='啟用高度自我意識的採樣策略，模擬深度思考 REQUIRE TESTING'
                    )
            
                    with gr.Column(visible=generate_params.get('panoptic_consciousness', False)) as panoptic_params:
                            shared.gradio['panoptic_target_entropy'] = gr.Slider(
                                0.0, 1.0,
                                value=generate_params.get('panoptic_target_entropy', 0.7),
                                step=0.01,
                                label='Target Normalized Entropy (Δ)',
                                info='目標熵值，用於衡量生成的多樣性和混亂程度。'
                                )
                            shared.gradio['panoptic_sensitivity'] = gr.Slider(
                                0.0, 5.0,
                                value=generate_params.get('panoptic_sensitivity', 1.0),
                                step=0.1,
                                label='Sensitivity (λ)',
                                info='決定模型對熵偏差的反應程度。'
                                )
                            shared.gradio['panoptic_reflection_rate'] = gr.Slider(
                                0.0, 1.0,
                                value=generate_params.get('panoptic_reflection_rate', 0.1),
                                step=0.01,
                                label='Reflection Rate',
                                info='自我反思的速度，控制自意識累積的速率。'
                                )
                            shared.gradio['panoptic_awakening_threshold'] = gr.Slider(
                                0.0, 10.0,
                                value=generate_params.get('panoptic_awakening_threshold', 5.0),
                                step=0.1,
                                label='Awakening Threshold',
                                info='當自我意識累積超過此閾值時，模型進入觉醒模式。'
                                )
                            shared.gradio['panoptic_awakening_multiplier'] = gr.Slider(
                                1.0, 10.0,
                                value=generate_params.get('panoptic_awakening_multiplier', 3.0),
                                step=0.1,
                                label='Awakening Multiplier',
                                info='觉醒模式下調整效果的放大倍數。'
                                )
                            shared.gradio['panoptic_reflection_iterations'] = gr.Slider(
                                1, 10,
                                value=generate_params.get('panoptic_reflection_iterations', 3),
                                step=1,
                                label='Reflection Iterations',
                                info='每次生成中的反思迭代次數。'
                                )
                            shared.gradio['panoptic_temporal_memory_factor'] = gr.Slider(
                                0.0, 5.0,
                                value=generate_params.get('panoptic_temporal_memory_factor', 1.0),
                                step=0.1,
                                label='Temporal Memory Factor (γ)',
                                info='根據上下文中 token 的頻率調整懲罰。'
                                )
                            shared.gradio['panoptic_ethical_modulation'] = gr.Slider(
                                0.0, 5.0,
                                value=generate_params.get('panoptic_ethical_modulation', 1.0),
                                step=0.1,
                                label='Ethical Modulation (δ)',
                                info='對伦理偏差進行懲罰的強度。'
                                )
                            shared.gradio['panoptic_target_ethics'] = gr.Slider(
                                0.0, 1.0,
                                value=generate_params.get('panoptic_target_ethics', 0.8),
                                step=0.01,
                                label='Target Ethics',
                                info='目標倫理值，衡量生成內容的倫理標準。'
                                        )
                            shared.gradio['panoptic_workspace_decay'] = gr.Slider(
                                0.0, 1.0,
                                value=generate_params.get('panoptic_workspace_decay', 0.9),
                                step=0.01,
                                label='Workspace Decay',
                                info='工作區狀態更新的速度，控制全局工作區的融聚程度。'
                                    )
            
            # 綁定 Checkbox 的狀態到參數組的可見性
                            shared.gradio['panoptic_consciousness'].change(
                                lambda x: gr.update(visible=x),
                                inputs=[shared.gradio['panoptic_consciousness']],
                                outputs=[panoptic_params]
                                )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Sephirotic / Qliphotic Sampling")
                            shared.gradio['sephirotic_emanation'] = gr.Checkbox(value=generate_params.get('sephirotic_emanation', False), label='Sephirotic Emanation')
                            shared.gradio['seph_penalties'] = gr.Textbox(value="0,0,0,0,0,0,0,0,0,0", label='Seph Penalties')
                            shared.gradio['seph_scaling'] = gr.Slider(0.1, 5.0, value=generate_params.get('seph_scaling', 1.0), step=0.1, label='Seph Scaling')
                            shared.gradio['seph_phase'] = gr.Slider(-3.14, 3.14, value=generate_params.get('seph_phase', 0.0), step=0.01, label='Seph Phase')
                            shared.gradio['scriptural_weights'] = gr.Textbox(value="0,0,0,0,0,0,0,0,0,0", label='Scriptural Weights')
                        with gr.Column():
                            shared.gradio['qliphotic_inversion'] = gr.Checkbox(value=generate_params.get('qliphotic_inversion', False), label='Qliphotic Inversion')
                            shared.gradio['qliph_penalties'] = gr.Textbox(value="0,0,0,0,0,0,0,0,0,0", label='Qliph Penalties')
                            shared.gradio['qliph_scaling'] = gr.Slider(0.1, 5.0, value=generate_params.get('qliph_scaling', 1.0), step=0.1, label='Qliph Scaling')
                            shared.gradio['qliph_phase'] = gr.Slider(-3.14, 3.14, value=generate_params.get('qliph_phase', 0.0), step=0.01, label='Qliph Phase')
                            shared.gradio['scriptural_bonus'] = gr.Textbox(value="0,0,0,0,0,0,0,0,0,0", label='Scriptural Bonus')
                        
                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Number(
                                precision=0, 
                                step=256, 
                                value=get_truncation_length(), 
                                label='Truncate the prompt up to this length', 
                                info='The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048.'
                            )
                            shared.gradio['seed'] = gr.Number(
                                value=shared.settings['seed'], 
                                label='Seed (-1 for random)'
                            )
                            shared.gradio['sampler_priority'] = gr.Textbox(
                                value=generate_params['sampler_priority'], 
                                lines=12, 
                                label='Sampler priority', 
                                info='Parameter names separated by new lines or commas.', 
                                elem_classes=['add_scrollbar']
                            )
                            shared.gradio['custom_stopping_strings'] = gr.Textbox(
                                lines=2, 
                                value=shared.settings["custom_stopping_strings"] or None, 
                                label='Custom stopping strings', 
                                info='Written between "" and separated by commas.', 
                                placeholder='"\\n", "\\nYou:"'
                            )
                            shared.gradio['custom_token_bans'] = gr.Textbox(
                                value=shared.settings['custom_token_bans'] or None, 
                                label='Token bans', 
                                info='Token IDs to ban, separated by commas. The IDs can be found in the Default or Notebook tab.'
                            )
                            shared.gradio['negative_prompt'] = gr.Textbox(
                                value=shared.settings['negative_prompt'], 
                                label='Negative prompt', 
                                info='For CFG. Only used when guidance_scale is different than 1.', 
                                lines=3, 
                                elem_classes=['add_scrollbar']
                            )
                            shared.gradio['dry_sequence_breakers'] = gr.Textbox(
                                value=generate_params['dry_sequence_breakers'], 
                                label='dry_sequence_breakers', 
                                info='Tokens across which sequence matching is not continued. Specified as a comma-separated list of quoted strings.'
                            )
                            with gr.Row() as shared.gradio['grammar_file_row']:
                                shared.gradio['grammar_file'] = gr.Dropdown(
                                    value='None', 
                                    choices=utils.get_available_grammars(), 
                                    label='Load grammar from file (.gbnf)', 
                                    elem_classes='slim-dropdown'
                                )
                                ui.create_refresh_button(
                                    shared.gradio['grammar_file'], 
                                    lambda: None, 
                                    lambda: {'choices': utils.get_available_grammars()}, 
                                    'refresh-button', 
                                    interactive=not mu
                                )
                                shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                                shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['grammar_string'] = gr.Textbox(
                                value='', 
                                label='Grammar', 
                                lines=16, 
                                elem_classes=['add_scrollbar', 'monospace']
                            )
        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(
        loaders.blacklist_samplers, 
        gradio('filter_by_loader', 'dynamic_temperature'), 
        gradio(loaders.list_all_samplers()), 
        show_progress=False
    )
    shared.gradio['preset_menu'].change(
        ui.gather_interface_values, 
        gradio(shared.input_elements), 
        gradio('interface_state')
    ).then(
        presets.load_preset_for_ui, 
        gradio('preset_menu', 'interface_state'), 
        gradio('interface_state') + gradio(presets.presets_params()), 
        show_progress=False
    )

    shared.gradio['random_preset'].click(
        ui.gather_interface_values, 
        gradio(shared.input_elements), 
        gradio('interface_state')
    ).then(
        presets.random_preset, 
        gradio('interface_state'), 
        gradio('interface_state') + gradio(presets.presets_params()), 
        show_progress=False
    )

    shared.gradio['grammar_file'].change(
        load_grammar, 
        gradio('grammar_file'), 
        gradio('grammar_string'), 
        show_progress=False
    )
    shared.gradio['dynamic_temperature'].change(
        lambda x: [gr.update(visible=x)] * 3, 
        gradio('dynamic_temperature'), 
        gradio('dynatemp_low', 'dynatemp_high', 'dynatemp_exponent'), 
        show_progress=False
    )


def get_truncation_length():
    if 'max_seq_len' in shared.provided_arguments or shared.args.max_seq_len != shared.args_defaults.max_seq_len:
        return shared.args.max_seq_len
    elif 'n_ctx' in shared.provided_arguments or shared.args.n_ctx != shared.args_defaults.n_ctx:
        return shared.args.n_ctx
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r', encoding='utf-8').read()
    else:
        return ''
