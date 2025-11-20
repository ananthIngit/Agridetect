import { useState, useEffect, useCallback, useRef } from "react";
import { Send, Bot, User, CornerUpLeft, Loader2 } from "lucide-react";

// Placeholder type declarations for external components (assuming basic props)
interface NavLinkProps {
    to: string;
    className: string;
    children: React.ReactNode;
}
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    className?: string;
    variant?: string;
    size?: string;
    children: React.ReactNode;
}
interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    className?: string;
    children: React.ReactNode;
}
// Placeholder for external components/hooks (must be included for single-file compilation)
const Button: React.FC<ButtonProps> = (props) => <button {...props} className={"p-3 rounded-lg " + props.className}>{props.children}</button>;
const Card: React.FC<CardProps> = (props) => <div {...props} className={"bg-white p-4 rounded-xl shadow-lg " + props.className}>{props.children}</div>;
const NavLink: React.FC<NavLinkProps> = (props) => <a href={props.to} className={props.className}>{props.children}</a>;

// --- TYPESCRIPT INTERFACES ---
interface Message {
    role: 'user' | 'bot';
    content: string;
}
// -----------------------------

// --- Configuration ---
// NOTE: API KEY IS HARDCODED FOR TESTING. REPLACE WITH YOUR ACTUAL KEY.
const GEMINI_API_KEY = "AIzaSyDXJN8egEzTG2WHS0HWq2SsQIkBTj6fg4k"; 
const MODEL_NAME = "gemini-2.5-flash-preview-09-2025";
const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_NAME}:generateContent?key=${GEMINI_API_KEY}`;
// ---------------------

// Simplified way to get URL parameters in a single-file React context
const useQuery = (): URLSearchParams => {
    return new URLSearchParams(window.location.search);
};

const Chat: React.FC = () => {
    const query = useQuery();
    const initialDisease: string | null = query.get("disease")?.replace(/-/g, ' ') || null;
    
    const [inputMessage, setInputMessage] = useState<string>("");
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };
    useEffect(scrollToBottom, [messages]);

    const systemInstruction: string = `You are AgriBot, an expert agricultural assistant specializing in plant disease diagnosis and treatment. Your tone should be encouraging, knowledgeable, and professional. Provide concise, actionable advice.`;

    useEffect(() => {
        if (initialDisease && messages.length === 0) {
            setMessages([
                { 
                    role: 'bot', 
                    content: `Welcome! I'm AgriBot. I see you're dealing with **${initialDisease}**. Before we begin, are your crops high-yield or organic, so I can tailor the best treatment advice?` 
                }
            ]);
        } else if (!initialDisease && messages.length === 0) {
             setMessages([
                { 
                    role: 'bot', 
                    content: `Welcome! I'm AgriBot. I can help you with plant health, diagnosis, and treatment. Ask me anything about your crops!` 
                }
            ]);
        }
    }, [initialDisease]);

    // Function to handle API call with retries
    const generateContentWithRetry = useCallback(async (payload: any): Promise<string> => {
        const MAX_RETRIES = 5;
        const INITIAL_DELAY = 1000; // 1 second
        
        for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (response.ok) {
                    const result = await response.json();
                    return result.candidates?.[0]?.content?.parts?.[0]?.text || "I apologize, I could not generate a response at this time.";
                } else if (response.status === 429 && attempt < MAX_RETRIES - 1) {
                    const delay = INITIAL_DELAY * (2 ** attempt);
                    await new Promise(resolve => setTimeout(resolve, delay));
                } else {
                    const errorBody = await response.json();
                    throw new Error(`API Error ${response.status}: ${JSON.stringify(errorBody)}`);
                }
            } catch (error) {
                if (attempt === MAX_RETRIES - 1) {
                    throw new Error(`Failed to communicate with LLM after ${MAX_RETRIES} attempts. ${error instanceof Error ? error.message : "Unknown error"}`);
                }
                const delay = INITIAL_DELAY * (2 ** attempt);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        return "Failed to communicate with the LLM.";
    }, []);

    const sendMessage = useCallback(async (text: string) => {
        if (!text.trim() || isLoading) return;

        const userMessage: Message = { role: 'user', content: text };
        setMessages(prev => [...prev, userMessage]);
        setInputMessage("");
        setIsLoading(true);

        const chatHistory = messages.map(msg => ({ 
            role: msg.role === 'bot' ? 'model' : 'user', 
            parts: [{ text: msg.content.replace(/\*\*/g, '') }] 
        }));
        
        chatHistory.push({ role: 'user', parts: [{ text }] });

        const payload = {
            contents: chatHistory,
            systemInstruction: {
                parts: [{ text: systemInstruction }]
            },
            // CRITICAL FIX: Adding Google Search Tool for connection stability
            tools: [{ "google_search": {} }],
        };

        try {
            const botText = await generateContentWithRetry(payload);

            const botResponse: Message = {
                role: 'bot',
                content: botText
            };
            setMessages(prev => [...prev, botResponse]);

        } catch (error) {
            console.error("Gemini API Error:", error);
            const errorMessage: Message = { 
                role: 'bot', 
                content: "Sorry, I am having trouble connecting to the AI assistant right now. Please check your network or try refreshing the page." 
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    }, [isLoading, messages, generateContentWithRetry]);

    const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !isLoading) {
            sendMessage(inputMessage);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-background to-muted font-sans">
            <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
                <div className="container mx-auto px-4 py-4 flex items-center justify-between">
                    <NavLink to="/" className="text-2xl font-bold text-primary">
                        AgriDetect
                    </NavLink>
                    <div className="flex gap-4">
                        <NavLink to="/" className="text-foreground hover:text-primary transition-colors">Home</NavLink>
                        <NavLink to="/detection" className="text-foreground hover:text-primary transition-colors">Detection</NavLink>
                        <NavLink to="/chat" className="text-primary font-semibold">AI Assistant</NavLink>
                    </div>
                </div>
            </nav>

            <main className="container mx-auto px-4 py-12">
                <div className="max-w-4xl mx-auto space-y-8">
                    <div className="flex items-center space-x-4 mb-8">
                        <NavLink to="/detection" className=""> {/* FIX: Added missing className */}
                            <Button variant="outline" className="text-muted-foreground hover:bg-muted/70">
                                <CornerUpLeft className="w-4 h-4 mr-2" /> Back to Detection
                            </Button>
                        </NavLink>
                        <h1 className="text-3xl font-bold text-foreground">
                            AI Treatment Assistant
                        </h1>
                    </div>

                    <Card className="p-0 h-[70vh] flex flex-col shadow-2xl">
                        {/* Chat History */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-muted/20">
                            {messages.length === 0 && (
                                <div className="text-center text-muted-foreground p-10">
                                    {initialDisease ? 
                                        `Starting consultation for ${initialDisease}...` :
                                        "Welcome! Ask me anything about plant diseases and care."
                                    }
                                </div>
                            )}
                            {messages.map((msg, index) => (
                                <div 
                                    key={index} 
                                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div className="flex items-start max-w-[80%] space-x-2">
                                        {msg.role === 'bot' && <Bot className="w-5 h-5 flex-shrink-0 mt-1 text-accent" />}
                                        <div className={`p-3 rounded-xl shadow-md 
                                            ${msg.role === 'user' 
                                                ? 'bg-primary text-primary-foreground rounded-br-none' 
                                                : 'bg-card border border-border text-foreground rounded-tl-none'}`}
                                        >
                                            <p className="whitespace-pre-wrap">{msg.content}</p>
                                        </div>
                                        {msg.role === 'user' && <User className="w-5 h-5 flex-shrink-0 mt-1 text-primary" />}
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-card border border-border text-foreground p-3 rounded-xl rounded-tl-none flex items-center space-x-2 shadow-md">
                                        <Loader2 className="w-5 h-5 animate-spin text-accent" />
                                        <p className="animate-pulse">AgriBot is thinking...</p>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} /> {/* Scroll target */}
                        </div>

                        {/* Input Area */}
                        <div className="p-4 border-t bg-card flex items-center">
                            <input
                                type="text"
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder="Ask about treatment, prevention, or symptoms..."
                                disabled={isLoading}
                                className="flex-1 p-3 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition duration-200 disabled:bg-muted/50 mr-2"
                            />
                            <Button
                                onClick={() => sendMessage(inputMessage)}
                                disabled={!inputMessage.trim() || isLoading}
                                className="bg-primary hover:bg-primary/90 text-primary-foreground disabled:bg-primary/40"
                            >
                                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                            </Button>
                        </div>
                    </Card>
                </div>
            </main>
        </div>
    );
};

export default Chat;