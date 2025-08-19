import {
  customProvider,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from 'ai';
import { xai } from '@ai-sdk/xai';
import {
  artifactModel,
  chatModel,
  reasoningModel,
  titleModel,
} from './models.test';
import { isTestEnvironment } from '../constants';
import { createHuggingFaceModel, FREE_MODELS } from './huggingface';

export const myProvider = isTestEnvironment
  ? customProvider({
      languageModels: {
        'chat-model': chatModel,
        'chat-model-reasoning': reasoningModel,
        'title-model': titleModel,
        'artifact-model': artifactModel,
      },
    })
  : customProvider({
      languageModels: {
        // Use free Hugging Face models
        'chat-model': createHuggingFaceModel({
          modelId: FREE_MODELS.MICROSOFT_DIALO_GPT,
        }),
        'chat-model-reasoning': createHuggingFaceModel({
          modelId: FREE_MODELS.FLAN_T5_BASE,
        }),
        'title-model': createHuggingFaceModel({
          modelId: FREE_MODELS.DISTIL_GPT2,
        }),
        'artifact-model': createHuggingFaceModel({
          modelId: FREE_MODELS.FLAN_T5_BASE,
        }),
      },
      imageModels: {
        'small-model': xai.imageModel('grok-2-image'),
      },
    });
