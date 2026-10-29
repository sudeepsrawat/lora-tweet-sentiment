# lora-tweet-sentiment
Fine-tuning Mistral-7B with LoRA (0.58% trainable params) for tweet sentiment analysis. Achieves 71.8% accuracy (+43 pts vs base) in 38 mins on a free Colab T4. Decomposes the performance lift into format compliance (+27 pts) and task learning (+16 pts). Built with Unsloth, PEFT, and Hugging Face.
