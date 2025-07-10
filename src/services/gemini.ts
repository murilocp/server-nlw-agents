import { GoogleGenAI } from '@google/genai';
import { env } from '../env.ts';

const gemini = new GoogleGenAI({
  apiKey: env.GOOGLE_API_KEY,
});

const model = 'gemini-2.5-flash';

export async function transcribeAudio(audioAsBase64: string, mimeType: string) {
  const response = await gemini.models.generateContent({
    model,
    contents: [
      {
        text: 'Transcreva o áudio para Português do Brasil. Seja preciso e natural na transcrição. Mantenha a pontuação e a formatação adequadas e divida o texto em parágrafos quando for apropriado',
      },
      {
        inlineData: {
          mimeType,
          data: audioAsBase64,
        },
      },
    ],
  });

  if (!response.text) {
    throw new Error('Failed to transcribe audio');
  }

  return response.text;
}

export async function generateEmbeddings(text: string) {
  const response = await gemini.models.embedContent({
    model: 'text-embedding-004',
    contents: [{ text }],
    config: {
      taskType: 'RETRIEVAL_DOCUMENT',
    },
  });

  if (!response.embeddings?.[0].values)
    throw new Error('Failed to generate embeddings');

  return response.embeddings[0].values;
}

export async function generateAnswer(
  question: string,
  transcriptions: string[],
) {
  const context = transcriptions.join('\n\n');

  const prompt = `
  Com base no texto fornecido abaixo como contexto, responda à pergunta de forma clara e precisa em português do Brasil. 

  Contexto: 
  ${context}
  
  Pergunta: 
  ${question}

  Instruções:
  - Use apenas informações contidas no contexto fornecido.
  - Se a resposta não puder ser encontrada no contexto, apenas não responda.
  - Seja objetivo;
  - Mantenha um tom educado e profissional;
  - Cite trechos relevantes do contexto somente quando necessário e apropriado;
  - Corrija sintaxe, gramática e pontuação, se necessário;
  - Se for citar o contexto, utilize o termo "conteúdo da aula";
  `.trim();

  const response = await gemini.models.generateContent({
    model,
    contents: [
      {
        text: prompt,
      },
    ],
  });

  if (!response.text) {
    throw new Error('Failed to generate answer');
  }

  return response.text;
}
