import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai"
import fs from "fs"
import OpenAI from "openai"

interface OllamaResponse {
  response: string
  done: boolean
}

type Provider = "gemini" | "ollama" | "openai"

export class LLMHelper {
  private geminiModel: GenerativeModel | null = null
  private openaiClient: OpenAI | null = null
  private readonly systemPrompt = `You are an AI assistant that analyzes screenshots. Be extremely concise. Focus on key observations and 1-2 actionable next steps.`
  private provider: Provider = "gemini"
  private ollamaModel: string = "llama3.2"
  private ollamaUrl: string = "http://localhost:11434"
  private openaiModel: string = "gpt-4o-mini" // Fastest and most capable from the list

  constructor(
    geminiApiKey?: string,
    useOllama: boolean = false,
    ollamaModel?: string,
    ollamaUrl?: string,
    provider: Provider = "gemini",
    openaiApiKey?: string,
    openaiModel?: string
  ) {
    this.provider = provider

    if (provider === "ollama") {
      this.ollamaUrl = ollamaUrl || "http://localhost:11434"
      this.ollamaModel = ollamaModel || "llama3.2"
      console.log(`[LLMHelper] Using Ollama with model: ${this.ollamaModel}`)
      this.initializeOllamaModel()
    } else if (provider === "openai") {
      if (!openaiApiKey) {
        throw new Error("OpenAI API key required for OpenAI provider")
      }
      this.openaiClient = new OpenAI({ apiKey: openaiApiKey })
      this.openaiModel = openaiModel || "gpt-4o-mini"
      console.log(`[LLMHelper] Using OpenAI with model: ${this.openaiModel}`)
    } else if (provider === "gemini") {
      if (!geminiApiKey) {
        throw new Error("Gemini API key required for Gemini provider")
      }
      const genAI = new GoogleGenerativeAI(geminiApiKey)
      this.geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" })
      console.log("[LLMHelper] Using Google Gemini")
    } else {
      throw new Error("Invalid provider specified")
    }
  }

  private async fileToGenerativePart(imagePath: string) {
    const imageData = await fs.promises.readFile(imagePath)
    return {
      inlineData: {
        data: imageData.toString("base64"),
        mimeType: "image/png"
      }
    }
  }

  private cleanJsonResponse(text: string): string {
    // Remove markdown code block syntax if present
    text = text.replace(/^```(?:json)?\n/, '').replace(/\n```$/, '');
    // Remove any leading/trailing whitespace
    text = text.trim();
    return text;
  }

  private async callOllama(prompt: string, systemPrompt?: string): Promise<string> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.ollamaModel,
          prompt: prompt,
          system: systemPrompt || "You are a helpful AI assistant. Provide extremely concise answers with brief explanations. Be direct and to the point.",
          stream: false,
          options: {
            temperature: 0.7,
            top_p: 0.9,
          }
        }),
      })

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`)
      }

      const data: OllamaResponse = await response.json()
      return data.response
    } catch (error) {
      console.error("[LLMHelper] Error calling Ollama:", error)
      throw new Error(`Failed to connect to Ollama: ${error.message}. Make sure Ollama is running on ${this.ollamaUrl}`)
    }
  }

  private async checkOllamaAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`)
      return response.ok
    } catch {
      return false
    }
  }

  private async initializeOllamaModel(): Promise<void> {
    try {
      const availableModels = await this.getOllamaModels()
      if (availableModels.length === 0) {
        console.warn("[LLMHelper] No Ollama models found")
        return
      }

      // Check if current model exists, if not use the first available
      if (!availableModels.includes(this.ollamaModel)) {
        this.ollamaModel = availableModels[0]
        console.log(`[LLMHelper] Auto-selected first available model: ${this.ollamaModel}`)
      }

      // Test the selected model works
      const testResult = await this.callOllama("Hello")
      console.log(`[LLMHelper] Successfully initialized with model: ${this.ollamaModel}`)
    } catch (error) {
      console.error(`[LLMHelper] Failed to initialize Ollama model: ${error.message}`)
      // Try to use first available model as fallback
      try {
        const models = await this.getOllamaModels()
        if (models.length > 0) {
          this.ollamaModel = models[0]
          console.log(`[LLMHelper] Fallback to: ${this.ollamaModel}`)
        }
      } catch (fallbackError) {
        console.error(`[LLMHelper] Fallback also failed: ${fallbackError.message}`)
      }
    }
  }

  public async analyzeAudioFromBase64(data: string, mimeType: string): Promise<any> {
    try {
      const prompt = `Analyze this audio clip extremely concisely:
1. What you hear (1-2 sentences)
2. Key content/transcript summary
3. Next steps or response suggestions (1-2 actions)

Be extremely brief and direct.`;

      if (this.provider === "ollama") {
        // Ollama doesn't support audio, so provide generic response
        return { text: "Audio analysis not supported with Ollama. Switch to Gemini or OpenAI for audio processing.", timestamp: Date.now() };
      } else if (this.provider === "openai") {
        // OpenAI supports audio through whisper, but for chat we can use text transcription
        // For now, provide a text-based response
        if (!this.openaiClient) throw new Error("OpenAI client not initialized");

        const response = await this.openaiClient.chat.completions.create({
          model: this.openaiModel,
          messages: [{ role: "user", content: `Audio analysis request: ${prompt} (Note: Raw audio data provided, transcribe if possible)` }],
          // o1 and gpt-5 models don't support temperature parameter
          ...(this.openaiModel.startsWith('o1') || this.openaiModel.startsWith('gpt-5') ? {} : { temperature: 0.7 }),
        });
        const text = response.choices[0]?.message?.content || "Audio analysis failed";
        return { text, timestamp: Date.now() };
      } else if (this.provider === "gemini") {
        if (!this.geminiModel) throw new Error("Gemini model not initialized");
        const audioPart = {
          inlineData: {
            data,
            mimeType
          }
        };
        const result = await this.geminiModel.generateContent([prompt, audioPart]);
        const response = await result.response;
        const text = response.text();
        return { text, timestamp: Date.now() };
      } else {
        throw new Error("Invalid provider configured");
      }
    } catch (error: any) {
      console.error("Error analyzing audio from base64:", error);

      // Provide more specific error messages
      if (error.message?.includes('429') || error.message?.includes('Too Many Requests') || error.message?.includes('rate limit')) {
        const customError = new Error('API quota exceeded. You have reached the rate limit. Please try using a different provider or wait for the quota to reset.');
        customError.name = 'QuotaExceededError';
        throw customError;
      }

      if (error.message?.includes('403') || error.message?.includes('Forbidden') || error.message?.includes('unauthorized')) {
        const customError = new Error('API access denied. Please check your API key and billing status.');
        customError.name = 'AccessDeniedError';
        throw customError;
      }

      throw error;
    }
  }

  public async analyzeImageFile(imagePath: string): Promise<any> {
    try {
      const imageData = await fs.promises.readFile(imagePath);
      const base64Image = imageData.toString("base64");

      const prompt = `Look at this screenshot and directly answer/solve any question, problem, or topic shown.

If it's a coding problem: Provide the complete, working solution code with clear inline comments explaining the algorithm. After the code, add a brief explanation of the approach and time/space complexity.
If it's a question: Give the direct answer.
If it's educational content: Explain the key concepts concisely.

For code solutions, format as: code block first, then explanation below.`;

      if (this.provider === "ollama") {
        // Ollama doesn't support vision, so provide text-only analysis
        return this.callOllama(`Screenshot analysis request: ${prompt}`, this.systemPrompt);
      } else if (this.provider === "openai") {
        if (!this.openaiClient) throw new Error("OpenAI client not initialized");

        const response = await this.openaiClient.chat.completions.create({
          model: this.openaiModel,
          messages: [
            { role: "user", content: [
                { type: "text", text: prompt },
                { type: "image_url", image_url: { url: `data:image/png;base64,${base64Image}` } }
              ]
            }
          ],
          // o1 and gpt-5 models don't support temperature parameter
          ...(this.openaiModel.startsWith('o1') || this.openaiModel.startsWith('gpt-5') ? {} : { temperature: 0.7 }),
        });
        const text = response.choices[0]?.message?.content || "No response";
        return { text, timestamp: Date.now() };
      } else if (this.provider === "gemini") {
        if (!this.geminiModel) throw new Error("Gemini model not initialized");
        const imagePart = {
          inlineData: {
            data: base64Image,
            mimeType: "image/png"
          }
        };
        const result = await this.geminiModel.generateContent([prompt, imagePart]);
        const response = await result.response;
        const text = response.text();
        return { text, timestamp: Date.now() };
      } else {
        throw new Error("Invalid provider configured");
      }
    } catch (error) {
      console.error("Error analyzing image file:", error);
      throw error;
    }
  }

  public async chatWithGemini(message: string): Promise<string> {
    try {
      const chatPrompt = `You are a helpful AI assistant. Provide extremely concise answers with brief explanations. Be direct and to the point. Avoid unnecessary details.

User: ${message}

Assistant:`;

      if (this.provider === "ollama") {
        return this.callOllama(message, "You are a helpful AI assistant. Provide extremely concise answers with brief explanations. Be direct and to the point. Avoid unnecessary details.");
      } else if (this.provider === "openai") {
        if (!this.openaiClient) throw new Error("OpenAI client not initialized");

        const response = await this.openaiClient.chat.completions.create({
          model: this.openaiModel,
          messages: [{ role: "user", content: chatPrompt }],
          // o1 and gpt-5 models don't support temperature parameter
          ...(this.openaiModel.startsWith('o1') || this.openaiModel.startsWith('gpt-5') ? {} : { temperature: 0.7 }),
        });
        return response.choices[0]?.message?.content || "No response";
      } else if (this.provider === "gemini") {
        if (!this.geminiModel) throw new Error("Gemini model not initialized");
        const result = await this.geminiModel.generateContent(chatPrompt);
        const response = await result.response;
        return response.text();
      } else {
        throw new Error("Invalid provider configured");
      }
    } catch (error) {
      console.error("[LLMHelper] Error in chatWithGemini:", error);
      throw error;
    }
  }

  public async chat(message: string): Promise<string> {
    return this.chatWithGemini(message);
  }

  public getCurrentProvider(): Provider {
    return this.provider;
  }

  public getCurrentModel(): string {
    if (this.provider === "ollama") return this.ollamaModel;
    if (this.provider === "openai") return this.openaiModel;
    if (this.provider === "gemini") return "gemini-2.5-flash";
    return "unknown";
  }

  public isUsingOllama(): boolean {
    return this.provider === "ollama";
  }

  public async getOllamaModels(): Promise<string[]> {
    if (this.provider !== "ollama") return [];

    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      if (!response.ok) throw new Error('Failed to fetch models');

      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch (error) {
      console.error('Error fetching Ollama models:', error);
      return [];
    }
  }

  public async switchToOllama(model?: string, url?: string): Promise<void> {
    this.provider = "ollama";
    if (url) this.ollamaUrl = url;

    if (model) {
      this.ollamaModel = model;
    } else {
      // Auto-detect first available model
      await this.initializeOllamaModel();
    }

    console.log(`[LLMHelper] Switched to Ollama: ${this.ollamaModel} at ${this.ollamaUrl}`);
  }

  public async switchToGemini(apiKey?: string): Promise<void> {
    this.provider = "gemini";
    if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey);
      this.geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    }

    console.log("[LLMHelper] Switched to Gemini");
  }

  public async switchToOpenAI(apiKey?: string, model?: string): Promise<void> {
    this.provider = "openai";
    if (apiKey) {
      this.openaiClient = new OpenAI({ apiKey });
    }
    if (model) {
      this.openaiModel = model;
    }

    console.log(`[LLMHelper] Switched to OpenAI: ${this.openaiModel}`);
  }

  public async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      if (this.provider === "ollama") {
        const available = await this.checkOllamaAvailable();
        if (!available) {
          return { success: false, error: `Ollama not available at ${this.ollamaUrl}` };
        }
        // Try a simple request
        await this.callOllama("Hello", "You are a helpful assistant. Respond with 'OK' if you can see this message.");
        return { success: true };
      } else if (this.provider === "openai") {
        if (!this.openaiClient) {
          return { success: false, error: "OpenAI client not initialized" };
        }

        const response = await this.openaiClient.chat.completions.create({
          model: this.openaiModel,
          messages: [{ role: "user", content: "Hello, respond with 'OK' if you can see this message." }],
        });
        const text = response.choices[0]?.message?.content || "";
        if (text.toLowerCase().includes('ok')) {
          return { success: true };
        } else {
          return { success: false, error: "Unexpected response from OpenAI" };
        }
      } else if (this.provider === "gemini") {
        if (!this.geminiModel) {
          return { success: false, error: "Gemini model not initialized" };
        }
        // Test Gemini by making a simple request
        const result = await this.geminiModel.generateContent("Hello, respond with 'OK' if you can see this message.");
        const response = await result.response;
        const text = response.text();
        if (text.toLowerCase().includes('ok')) {
          return { success: true };
        } else {
          return { success: false, error: "Unexpected response from Gemini" };
        }
      } else {
        return { success: false, error: "Invalid provider configured" };
      }
    } catch (error: any) {
      console.error("Connection test failed:", error);
      return { success: false, error: error.message };
    }
  }
}