import { HfInference } from '@huggingface/inference';
import type { LanguageModelV1 } from '@ai-sdk/provider';

interface HuggingFaceModelConfig {
  modelId: string;
  apiKey?: string;
}

export function createHuggingFaceModel(config: HuggingFaceModelConfig): LanguageModelV1 {
  const hf = new HfInference(config.apiKey || process.env.HUGGINGFACE_API_KEY);

  return {
    specificationVersion: 'v1',
    provider: 'huggingface',
    modelId: config.modelId,
    defaultObjectGenerationMode: undefined,

    async doGenerate(options) {
      try {
        // Convert messages to a single prompt
        const prompt = options.prompt
          .map((msg) => {
            if (msg.role === 'user') return `Human: ${msg.content[0]?.text || ''}`;
            if (msg.role === 'assistant') return `Assistant: ${msg.content[0]?.text || ''}`;
            if (msg.role === 'system') return `System: ${msg.content[0]?.text || ''}`;
            return '';
          })
          .filter(Boolean)
          .join('\n') + '\nAssistant:';

        const response = await hf.textGeneration({
          model: config.modelId,
          inputs: prompt,
          parameters: {
            max_new_tokens: Math.min(options.maxTokens || 150, 512),
            temperature: options.temperature || 0.7,
            return_full_text: false,
            do_sample: true,
            top_p: 0.95,
          },
        });

        return {
          text: response.generated_text?.trim() || 'I apologize, but I could not generate a response.',
          finishReason: 'stop' as const,
          usage: {
            promptTokens: 0,
            completionTokens: 0,
          },
        };
      } catch (error) {
        console.error('Hugging Face API error:', error);
        return {
          text: 'I apologize, but I encountered an error. Please try again.',
          finishReason: 'stop' as const,
          usage: {
            promptTokens: 0,
            completionTokens: 0,
          },
        };
      }
    },

    async doStream(options) {
      const result = await this.doGenerate(options);
      const text = result.text;
      const words = text.split(' ');
      
      return {
        stream: (async function* () {
          for (let i = 0; i < words.length; i++) {
            const word = words[i] + (i < words.length - 1 ? ' ' : '');
            yield {
              type: 'text-delta' as const,
              textDelta: word,
            };
            await new Promise(resolve => setTimeout(resolve, 30));
          }
          
          yield {
            type: 'finish' as const,
            finishReason: 'stop' as const,
            usage: {
              promptTokens: 0,
              completionTokens: 0,
            },
          };
        })(),
        rawCall: { rawPrompt: '', rawSettings: {} },
      };
    },
  };
}

// Free models you can use
export const FREE_MODELS = {
  MICROSOFT_DIALO_GPT: 'microsoft/DialoGPT-large',
  FACEBOOK_BLENDER: 'facebook/blenderbot-400M-distill',
  GPT2: 'gpt2',
  DISTIL_GPT2: 'distilgpt2',
  FLAN_T5_SMALL: 'google/flan-t5-small',
  FLAN_T5_BASE: 'google/flan-t5-base',
};
