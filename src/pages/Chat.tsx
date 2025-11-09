import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Bot, User, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { NavLink } from "@/components/NavLink";
import { useToast } from "@/hooks/use-toast";
import { useSearchParams } from "react-router-dom";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: { uri: string; title: string }[];
}

const Chat = () => {
  const [searchParams] = useSearchParams();
  // Get initial disease name from the URL query parameter (e.g., /chat?disease=early-blight)
  const initialDisease = searchParams.get('disease');
  
  const initialMessage: Message[] = [
    {
      role: "assistant",
      content: initialDisease
        ? `I see you detected **${initialDisease.toUpperCase().replace(/-/g, ' ')}**! I'm your AI farming assistant. I can immediately provide treatment and management advice for this. What would you like to know about ${initialDisease.replace(/-/g, ' ')}?`
        : "Hello! I'm your AI farming assistant. I can help you with plant diseases, crop management, soil health, and any agriculture-related questions. How can I assist you today?",
    }
  ];

  const [messages, setMessages] = useState<Message[]>(initialMessage);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getSystemPrompt = useCallback(() => {
    
    // This new prompt defines a dual role.
    const basePrompt = `You are a helpful, friendly, and knowledgeable AI assistant.
- Your primary expertise is as an agricultural farming assistant. Provide practical, concise, and scientifically grounded advice on crop management, soil health, and plant disease.
- However, you are also a general-purpose AI, like Gemini. You can chat about any topic, answer general questions, help with code, or just have a normal, friendly conversation.
- Be conversational and natural in your tone.
- For farming or scientific questions, base your answers on real-time web search results to ensure accuracy.
- For general chat, just be your helpful self.`;

    // If a disease was detected, add it as *context* for the start of the chat.
    if (initialDisease) {
      return `${basePrompt}
- The user was just looking at a plant with **${initialDisease.replace(/-/g, ' ')}**. Be prepared to give expert advice on it, but feel free to chat about anything else they want.`;
    }

    // This is the default prompt if no disease was detected.
    return basePrompt;
  }, [initialDisease]);

  // --- Gemini API Integration Logic ---
  const geminiFetch = useCallback(async (userQuery: string) => {
    const systemPrompt = getSystemPrompt();
    const apiKey = "AIzaSyDXJN8egEzTG2WHS0HWq2SsQIkBTj6fg4k"; // Canvas environment will provide the API key
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;
    
    const payload = {
        contents: [{ parts: [{ text: userQuery }] }],
        tools: [{ "google_search": {} }],
        systemInstruction: {
            parts: [{ text: systemPrompt }]
        },
    };

    let responseJson: any = null;
    let attempts = 0;
    const maxAttempts = 3;
    
    // Exponential backoff retry loop
    while (attempts < maxAttempts) {
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                responseJson = await response.json();
                break; 
            } else {
                throw new Error(`API returned status ${response.status}`);
            }
        } catch (error) {
            console.error(`Attempt ${attempts + 1} failed:`, error);
            attempts++;
            if (attempts < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempts) * 1000));
            } else {
                throw new Error("Failed to get response from AI after multiple retries.");
            }
        }
    }
    
    if (!responseJson) {
        throw new Error("Could not parse AI response.");
    }

    const candidate = responseJson.candidates?.[0];
    const text = candidate?.content?.parts?.[0]?.text || "Sorry, I couldn't generate a response right now.";
    
    let sources: { uri: string; title: string }[] = [];
    const groundingMetadata = candidate?.groundingMetadata;
    if (groundingMetadata && groundingMetadata.groundingAttributions) {
        sources = groundingMetadata.groundingAttributions
            .map((attribution: any) => ({
                uri: attribution.web?.uri,
                title: attribution.web?.title,
            }))
            .filter(source => source.uri && source.title);
    }
    
    return { text, sources };

  }, [getSystemPrompt]);


  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const { text, sources } = await geminiFetch(userMessage);
      setMessages(prev => [...prev, { role: "assistant", content: text, sources }]);
      
    } catch (error) {
      toast({
        title: "AI Error",
        description: (error as Error).message,
        variant: "destructive",
      });
      setMessages(prev => [...prev, { role: "assistant", content: "I apologize, but I encountered an error while trying to generate advice. Please try refreshing or checking your input." }]);
      
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };
  
  const renderSources = (sources?: { uri: string; title: string }[]) => {
    if (!sources || sources.length === 0) return null;
    
    const uniqueSources = Array.from(new Map(sources.map(s => [s.uri, s])).values());
    
    return (
      <div className="mt-2 text-xs text-muted-foreground">
        <p className="font-semibold mb-1">Sources:</p>
        <ul className="list-disc list-inside space-y-0.5">
          {uniqueSources.map((source, i) => (
            <li key={i}>
              <a 
                href={source.uri} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="hover:underline text-primary"
                title={source.title}
              >
                {source.title.length > 50 ? `${source.title.substring(0, 50)}...` : source.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex flex-col">
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <NavLink to="/" className="text-2xl font-bold text-primary">
            AgriDetect
          </NavLink>
          <div className="flex gap-4">
            <NavLink to="/" className="text-foreground hover:text-primary transition-colors">
              Home
            </NavLink>
            <NavLink to="/detection" className="text-foreground hover:text-primary transition-colors">
              Detection
            </NavLink>
            <NavLink to="/chat" className="text-primary font-semibold">
              AI Assistant
            </NavLink>
          </div>
        </div>
      </nav>

      <main className="flex-1 container mx-auto px-4 py-8 flex flex-col max-w-4xl">
        <div className="text-center mb-6">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-2">
            AI Farming Assistant
          </h1>
          <p className="text-muted-foreground">
            Ask me anything about agriculture, plant care, or disease management
          </p>
        </div>

        <Card className="flex-1 flex flex-col shadow-card overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-3 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {message.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5 text-primary-foreground" />
                  </div>
                )}
                
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-foreground"
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                  {message.sources && renderSources(message.sources)}
                </div>

                {message.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5 text-secondary-foreground" />
                  </div>
                )}
              </div>
            ))}
            
            {isLoading && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-primary-foreground" />
                </div>
                <div className="bg-muted rounded-lg p-4">
                  <Loader2 className="w-4 h-4 text-primary animate-spin" />
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-border p-4">
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about plant diseases, farming tips, or agriculture..."
                className="flex-1"
                disabled={isLoading}
              />
              <Button
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                className="bg-primary hover:bg-primary/90 text-primary-foreground"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </Card>
      </main>
    </div>
  );
};

export default Chat;