// Copyright 2026 The AI Edge LiteRT Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "SceneDelegate.h"

#import "AppDelegate.h"

@implementation SceneDelegate

- (void)scene:(UIScene*)scene
    willConnectToSession:(UISceneSession*)session
                 options:(UISceneConnectionOptions*)connectionOptions {
  UIViewController* controller = [[UIViewController alloc] init];
  UITextView* textView = [[UITextView alloc] initWithFrame:controller.view.bounds];
  textView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
  textView.editable = NO;
  textView.font = [UIFont monospacedSystemFontOfSize:11 weight:UIFontWeightRegular];
  [controller.view addSubview:textView];
  self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene*)scene];
  self.window.rootViewController = controller;
  [self.window makeKeyAndVisible];
  AppDelegate* app = (AppDelegate*)UIApplication.sharedApplication.delegate;
  app.textView = textView;
  textView.text = app.logText ?: @"benchmark_model running...";
}

@end
