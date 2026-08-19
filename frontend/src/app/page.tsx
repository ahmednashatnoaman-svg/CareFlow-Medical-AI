"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, BookOpen, Stethoscope, Search, FileText, ChevronRight, ShieldAlert } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { EvaluationDashboard } from "@/components/EvaluationDashboard";

export default function Home() {
  const [mode, setMode] = useState<"triage" | "guidelines" | "evaluation">("triage");

  return (
    <div className="min-h-screen bg-background dark p-4 md:p-8 flex flex-col items-center">
      <header className="w-full max-w-6xl mb-8 flex justify-between items-center glass p-4 rounded-2xl">
        <div className="flex items-center gap-3">
          <div className="bg-primary/20 p-2 rounded-xl text-primary">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">CareFlow <span className="text-primary font-light">Medical AI</span></h1>
            <p className="text-sm text-muted-foreground">Dual-Engine Clinical Intelligence</p>
          </div>
        </div>
        
        <Tabs value={mode} onValueChange={(v) => setMode(v as any)} className="w-[500px]">
          <TabsList className="grid w-full grid-cols-3 glass">
            <TabsTrigger value="triage" className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary">
              <Stethoscope className="w-4 h-4 mr-2" /> Triage
            </TabsTrigger>
            <TabsTrigger value="guidelines" className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary">
              <BookOpen className="w-4 h-4 mr-2" /> Guidelines
            </TabsTrigger>
            <TabsTrigger value="evaluation" className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary">
              <TrendingUp className="w-4 h-4 mr-2" /> Evaluation
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      <main className="flex-1 w-full max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        <AnimatePresence mode="wait">
          {mode === "triage" && (
            <motion.div 
              key="triage"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="lg:col-span-2 space-y-6"
            >
              <Card className="glass border-primary/20 h-[600px] flex flex-col">
                <CardHeader>
                  <CardTitle className="flex justify-between items-center">
                    Diagnostic Triage Interview
                    <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">SOCRATES Active</Badge>
                  </CardTitle>
                  <CardDescription>Graph RAG Engine exploring symptom phenotype space</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col">
                  <ScrollArea className="flex-1 pr-4">
                    <div className="space-y-4">
                      {/* AI Message */}
                      <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
                          <Activity className="w-4 h-4 text-primary" />
                        </div>
                        <div className="bg-card border border-border p-4 rounded-2xl rounded-tl-sm shadow-sm max-w-[85%] text-sm">
                          <p>Welcome. Please describe your primary symptom or reason for visit.</p>
                        </div>
                      </div>
                    </div>
                  </ScrollArea>
                </CardContent>
                <CardFooter className="pt-4 border-t border-border/50">
                  <div className="w-full relative">
                    <Input placeholder="Type symptom..." className="pr-12 bg-background/50 border-primary/20 h-12 rounded-xl" />
                    <Button size="icon" className="absolute right-1 top-1 h-10 w-10 rounded-lg">
                      <ChevronRight className="w-5 h-5" />
                    </Button>
                  </div>
                </CardFooter>
              </Card>
            </motion.div>
          )}

          {mode === "guidelines" && (
            <motion.div 
              key="guidelines"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="lg:col-span-2 space-y-6"
            >
              <Card className="glass border-primary/20 h-[600px] flex flex-col">
                <CardHeader>
                  <CardTitle className="flex justify-between items-center">
                    Guidelines Assistant
                    <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">Vector RAG</Badge>
                  </CardTitle>
                  <CardDescription>Grounded strictly in WHO, CDC, NICE, and USPSTF official guidelines</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col">
                  <ScrollArea className="flex-1 pr-4">
                    <div className="space-y-4 flex flex-col items-center justify-center h-full text-muted-foreground text-sm">
                      <ShieldAlert className="w-12 h-12 mb-4 opacity-50" />
                      <p>Ask a clinical question. All answers will cite direct sources.</p>
                    </div>
                  </ScrollArea>
                </CardContent>
                <CardFooter className="pt-4 border-t border-border/50">
                  <div className="w-full relative">
                    <Input placeholder="Ask about guidelines..." className="pr-12 bg-background/50 border-primary/20 h-12 rounded-xl" />
                    <Button size="icon" className="absolute right-1 top-1 h-10 w-10 rounded-lg">
                      <Search className="w-5 h-5" />
                    </Button>
                  </div>
                </CardFooter>
              </Card>
            </motion.div>
          )}

          {mode === "evaluation" && (
            <motion.div 
              key="evaluation"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="lg:col-span-2 space-y-6"
            >
              <EvaluationDashboard />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Right Sidebar: Evidence / Status */}
        <div className="space-y-6">
          <Card className="glass">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                <FileText className="w-4 h-4" /> System Telemetry
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span>Diagnostic Entropy</span>
                  <span className="font-mono">1.82</span>
                </div>
                <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-[70%]" />
                </div>
              </div>
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span>Confidence Margin</span>
                  <span className="font-mono">0.14</span>
                </div>
                <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 w-[30%]" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass flex-1 min-h-[300px]">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                <ShieldAlert className="w-4 h-4" /> Grounding Evidence
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-xs text-center py-8">
                Evidence citations and PDF exports will appear here during active sessions.
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
