// The made-up support tickets and questions of the benchmarks (about 80 to 120 words each).
import type { Question } from "../../src";

const who = ["Hi,", "Hello team,", "Good morning,", "To whom it may concern,"];
const what = [
  "I was charged twice for my last order (#{n}) and the second payment still has not been refunded after ten days.",
  "my parcel {n} has been stuck in transit for two weeks and the tracking page has not moved since Monday.",
  "the app crashes every time I open the settings page since the update to version 4.{n}; I already reinstalled it twice.",
  "I cannot log in to my account anymore: the password reset email never arrives, I checked the spam folder too.",
  "I would like to know whether your premium plan includes priority support and how invoices are sent each month.",
  "please cancel my subscription at the end of this billing period, I no longer need the service.",
];
const tail = [
  "This is the third time I write and nobody answers. If this is not fixed by Friday I will go to a competitor.",
  "No rush, whenever you have a moment. Thanks a lot for your help and have a nice day.",
  "I need this sorted today, our whole team is blocked and we have a client demo tomorrow morning.",
  "Could you also confirm my address is still correct? It is 12 rue des Lilas, Lyon. Thanks in advance.",
];
export const tickets = (n: number) => Array.from({ length: n }, (_, i) => `${who[i % 4]} ${what[(i * 7) % 6]!.replace("{n}", String(1000 + i))} ${tail[(i * 5) % 4]} Best regards, customer ${i}.`);
export const questions: Record<string, Question> = {
  topic: { type: "choice", instructions: "What is the main topic of the customer ticket?", criteria: { billing: "billing, payment, refund", delivery: "delivery, parcel", bug: "technical bug", account_access: "account, login, access", cancellation: "cancellation", info_request: "information request", other: "other" } },
  churn_risk: { type: "noul", instructions: "Is the customer at risk of leaving?" },
  urgency: { type: "score", instructions: "How urgent is the ticket?", criteria: ["low", "medium", "high"] },
};

