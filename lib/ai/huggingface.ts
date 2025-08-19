import { HfInference } from '@huggingface/inference';
import type { 
  LanguageModelV2, 
  LanguageModelV2CallOptions,
  LanguageModelV2Content,
  LanguageModelV2Usage,
  LanguageModelV2CallWarning,
  LanguageModelV2FinishReason
} from '@ai-sdk/provider';

interface HuggingFaceModelConfig {
  modelId: string;
  apiKey?: string;
}

export function createHuggingFaceModel(config: HuggingFaceModelConfig): LanguageModelV2 {
  const hf = new HfInference(config.apiKey || process.env.HUGGINGFACE_API_KEY);

  return {
    specificationVersion: 'v2',
    provider: 'huggingface',
    modelId: config.modelId,
    supportedUrls: {},

    async doGenerate(options: LanguageModelV2CallOptions) {
      try {
        // Convert messages to a single prompt
        const prompt = options.prompt
          .map((msg) => {
            if (msg.role === 'user') {
              // Handle user content - it's always an array
              return `Human: ${msg.content
                .map((part: any) => {
                  if (part.type === 'text') return part.text;
                  return '[File]';
                })
                .join(' ')}`;
            }
            if (msg.role === 'assistant') {
              // Handle assistant content - it's always an array
              return `Assistant: ${msg.content
                .map((part: any) => {
                  if (part.type === 'text') return part.text;
                  return '[File]';
                })
                .join(' ')}`;
            }
            if (msg.role === 'system') {
              // Handle system content - it's a string
              return `System: ${msg.content}`;
            }
            if (msg.role === 'tool') {
              // Handle tool content - it's an array
              return `Tool: ${(msg.content as any[])
                .map((part: any) => {
                  if (part.type === 'tool-result') return JSON.stringify(part.output);
                  return '[Tool Result]';
                })
                .join(' ')}`;
            }
            return '';
          })
          .filter(Boolean)
          .join('\n') + '\nAssistant:';

        const response = await hf.textGeneration({
          model: config.modelId,
          inputs: prompt,
          parameters: {
            max_new_tokens: Math.min(options.maxOutputTokens || 150, 512),
            temperature: options.temperature || 0.7,
            return_full_text: false,
            do_sample: true,
            top_p: options.topP || 0.95,
          },
        });

        const text = response.generated_text?.trim() || 'I apologize, but I could not generate a response.';

        const content: LanguageModelV2Content[] = [{ type: 'text', text }];
        const finishReason: LanguageModelV2FinishReason = 'stop';
        const usage: LanguageModelV2Usage = {
          inputTokens: undefined,
          outputTokens: undefined,
          totalTokens: undefined,
        };
        const warnings: LanguageModelV2CallWarning[] = [];

        return {
          content,
          finishReason,
          usage,
          warnings,
        };
      } catch (error) {
        console.error('Hugging Face API error:', error);
        
        const content: LanguageModelV2Content[] = [{ 
          type: 'text', 
          text: 'I apologize, but I encountered an error. Please try again.' 
        }];
        const finishReason: LanguageModelV2FinishReason = 'stop';
        const usage: LanguageModelV2Usage = {
          inputTokens: undefined,
          outputTokens: undefined,
          totalTokens: undefined,
        };
        const warnings: LanguageModelV2CallWarning[] = [];

        return {
          content,
          finishReason,
          usage,
          warnings,
        };
      }
    },

    async doStream(options: LanguageModelV2CallOptions) {
      // For free models, simulate streaming by generating and chunking
      try {
        const result = await this.doGenerate(options);
        const text = result.content[0]?.type === 'text' ? (result.content[0] as any).text : '';
        const words = text.split(' ');
        
        const stream = new ReadableStream({
          async start(controller) {
            try {
              for (let i = 0; i < words.length; i++) {
                const word = words[i] + (i < words.length - 1 ? ' ' : '');
                controller.enqueue({
                  type: 'text-delta' as const,
                  textDelta: word,
                });
                // Add small delay to simulate streaming
                await new Promise(resolve => setTimeout(resolve, 30));
              }
              
              controller.enqueue({
                type: 'finish' as const,
                finishReason: 'stop' as const,
                usage: {
                  inputTokens: undefined,
                  outputTokens: undefined,
                  totalTokens: undefined,
                },
              });
              
              controller.close();
            } catch (error) {
              controller.error(error);
            }
          },
        });

        return {
          stream,
        };
      } catch (error) {
        console.error('Hugging Face streaming error:', error);
        throw error;
      }
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
