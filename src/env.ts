import { z } from 'zod';

const envSchema = z.object({
  PORT: z.coerce.number().default(3333),
  DATABASE_URL: z.string().url().startsWith('postgresql://'),
  GOOGLE_API_KEY: z.string().min(32).max(100),
});

export const env = envSchema.parse(process.env);
